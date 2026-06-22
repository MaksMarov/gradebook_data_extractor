from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse


JOBS_DIR = Path(os.getenv("WEB_JOBS_DIR", "data/web_jobs"))

# pipeline | mock
PROCESSOR_MODE = os.getenv("WEB_PROCESSOR_MODE", "pipeline").strip().lower()
PIPELINE_DEBUG = os.getenv("WEB_PIPELINE_DEBUG", "1").strip().lower() not in {"0", "false", "no", "off"}
PIPELINE_CONCURRENCY = max(1, int(os.getenv("WEB_PIPELINE_CONCURRENCY", "1")))
MAX_UPLOAD_FILES = max(1, int(os.getenv("WEB_MAX_UPLOAD_FILES", "50")))
MAX_UPLOAD_SIZE_MB = max(1, int(os.getenv("WEB_MAX_UPLOAD_SIZE_MB", "25")))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {
    ("jpg" if item.strip().lower().lstrip(".") == "jpeg" else item.strip().lower().lstrip("."))
    for item in os.getenv("WEB_ALLOWED_IMAGE_EXTENSIONS", "jpg,jpeg,png,webp,bmp").split(",")
    if item.strip()
}

PIPELINE_DEVICE = os.getenv("PIPELINE_DEVICE", "auto").strip().lower()
PIPELINE_REQUIRE_GPU = os.getenv("PIPELINE_REQUIRE_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto").strip()
EASYOCR_GPU = os.getenv("EASYOCR_GPU", "auto").strip()

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gradebook Extractor API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_tasks: dict[str, asyncio.Task[Any]] = {}
_store_lock = asyncio.Lock()
_pipeline_semaphore = asyncio.Semaphore(PIPELINE_CONCURRENCY)


@app.on_event("startup")
async def startup() -> None:
    load_jobs_from_disk()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Gradebook Extractor API",
        "health": "/api/health",
        "docs": "/docs",
    }


@app.get("/api/health/live")
def health_live() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "backend",
        "time": now_iso(),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return build_health(check_dependencies=False)


@app.get("/api/health/ready")
def health_ready() -> dict[str, Any]:
    payload = build_health(check_dependencies=True)
    if payload["status"] != "ok":
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/api/health/ollama")
def health_ollama() -> dict[str, Any]:
    payload = check_ollama()
    if payload["status"] not in {"ok", "skipped"}:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/api/health/gpu")
def health_gpu() -> dict[str, Any]:
    payload = check_backend_gpu()
    if payload["status"] == "error":
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.post("/api/jobs")
async def create_jobs(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Не загружено ни одного файла")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Слишком много файлов за один раз: максимум {MAX_UPLOAD_FILES}",
        )

    prepared_uploads: list[dict[str, Any]] = []
    for upload in files:
        original_name = upload.filename or "image.jpg"
        extension = get_upload_extension(original_name)
        content = await upload.read()
        validate_upload(original_name, extension, upload.content_type, content)
        prepared_uploads.append(
            {
                "original_name": original_name,
                "extension": extension,
                "content": content,
            }
        )

    created: list[dict[str, Any]] = []

    async with _store_lock:
        for prepared in prepared_uploads:
            original_name = prepared["original_name"]
            extension = prepared["extension"]
            content = prepared["content"]

            job_id = uuid.uuid4().hex
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            input_path = job_dir / f"input.{extension}"
            input_path.write_bytes(content)

            now = now_iso()
            job = {
                "id": job_id,
                "filename": original_name,
                "safe_filename": safe_name(original_name),
                "size_bytes": len(content),
                "extension": extension,
                "input_path": str(input_path),
                "download_path": str(input_path),
                "result_path": None,
                "status": "queued",
                "status_text": "Ожидает",
                "student_number": None,
                "progress": 0,
                "message": "Файл принят в обработку",
                "created_at": now,
                "finished_at": None,
                "elapsed_sec": 0.0,
                "attempt": 0,
                "processor_mode": PROCESSOR_MODE,
                "error_code": None,
                "error_message": None,
                "artifacts": {},
                "artifact_paths": {},
                "pipeline_result": None,
            }

            _jobs[job_id] = job
            save_job(job)
            created.append(public_job(job))

    for job in created:
        schedule_processing(job["id"])

    return {"jobs": created}

@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    jobs = sorted(_jobs.values(), key=lambda item: item.get("created_at", ""), reverse=False)
    return {"jobs": [public_job(job) for job in jobs]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    return public_job(job)


@app.post("/api/jobs/retry-failed")
async def retry_failed() -> dict[str, Any]:
    retried: list[dict[str, Any]] = []

    async with _store_lock:
        for job in _jobs.values():
            if job["status"] != "error":
                continue

            job.update(
                {
                    "status": "queued",
                    "status_text": "Ожидает",
                    "student_number": None,
                    "progress": 0,
                    "message": "Файл поставлен на повторную обработку",
                    "finished_at": None,
                    "elapsed_sec": 0.0,
                    "download_path": job.get("input_path"),
                    "result_path": None,
                    "error_code": None,
                    "error_message": None,
                    "artifacts": {},
                    "artifact_paths": {},
                    "pipeline_result": None,
                    "processor_mode": PROCESSOR_MODE,
                }
            )
            save_job(job)
            retried.append(public_job(job))

    for job in retried:
        schedule_processing(job["id"])

    return {"jobs": retried}


@app.get("/api/jobs/{job_id}/download")
def download_job_result(job_id: str) -> FileResponse:
    job = require_job(job_id)

    if job["status"] != "ok" or not job.get("student_number"):
        raise HTTPException(status_code=409, detail="Job has no successful result")

    download_path = Path(job.get("download_path") or job.get("input_path") or "")
    if not download_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    extension = get_extension(download_path.name)
    return FileResponse(
        download_path,
        media_type=media_type_for_extension(extension),
        filename=f"{job['student_number']}.{extension}",
    )


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
def get_artifact(job_id: str, artifact_name: str) -> Response:
    job = require_job(job_id)
    artifact_name = artifact_name.lower().strip()

    artifact_path = resolve_artifact_path(job, artifact_name)
    if artifact_path and artifact_path.exists():
        return FileResponse(artifact_path, media_type=media_type_for_extension(get_extension(artifact_path.name)))

    labels = {
        "input": "Исходное изображение",
        "source": "Исходное изображение",
        "original": "Исходное изображение",
        "face": "Распознанное лицо",
        "result": "Результат: распознанное лицо",
        "anchor": "Промежуточный кроп: область номера",
        "recognized": "Промежуточный кроп: распознанный номер",
        "recognized_anchor": "Промежуточный кроп: распознанный номер",
        "number_crop": "Промежуточный кроп: распознанный номер",
        "yolo": "Промежуточный файл: детекция",
        "yolo_debug": "Промежуточный файл: детекция",
    }

    if artifact_name in labels:
        return svg_response(make_svg_placeholder(labels[artifact_name], job))

    raise HTTPException(status_code=404, detail="Artifact not found")


@app.get("/api/downloads/results.csv")
def download_results_csv() -> StreamingResponse:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["filename", "status", "student_number", "elapsed_sec", "message", "error_code"])

    for job in sorted(_jobs.values(), key=lambda item: item.get("created_at", "")):
        writer.writerow(
            [
                job.get("filename") or "",
                job.get("status") or "",
                job.get("student_number") or "",
                f"{float(job.get('elapsed_sec') or 0):.3f}" if job.get("elapsed_sec") else "",
                job.get("message") or "",
                job.get("error_code") or "",
            ]
        )

    content = buffer.getvalue().encode("utf-8-sig")
    headers = {"Content-Disposition": 'attachment; filename="results.csv"'}
    return StreamingResponse(io.BytesIO(content), media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/downloads/successful.zip")
def download_successful_zip() -> StreamingResponse:
    successful = [job for job in _jobs.values() if job.get("status") == "ok" and job.get("student_number")]

    memory = io.BytesIO()
    used_names: set[str] = set()

    with zipfile.ZipFile(memory, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for job in successful:
            download_path = Path(job.get("download_path") or job.get("input_path") or "")
            if not download_path.exists():
                continue

            extension = get_extension(download_path.name)
            archive_name = unique_archive_name(f"{job['student_number']}.{extension}", used_names)

            info = zipfile.ZipInfo(archive_name)
            info.compress_type = zipfile.ZIP_DEFLATED
            # Explicit UTF-8 file-name flag for Cyrillic names.
            info.flag_bits |= 0x800
            archive.writestr(info, download_path.read_bytes())

    memory.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="recognized_faces.zip"'}
    return StreamingResponse(memory, media_type="application/zip", headers=headers)


def build_health(*, check_dependencies: bool) -> dict[str, Any]:
    yolo_status = check_yolo_model()
    jobs_dir_status = check_jobs_dir()
    ollama_status = check_ollama() if check_dependencies else {"status": "not_checked"}
    backend_gpu_status = check_backend_gpu() if check_dependencies else {"status": "not_checked"}

    dependencies = {
        "jobs_dir": jobs_dir_status,
        "yolo_model": yolo_status,
        "ollama": ollama_status,
        "backend_gpu": backend_gpu_status,
    }

    dependency_values = [value["status"] for value in dependencies.values()]
    if check_dependencies and any(status == "error" for status in dependency_values):
        status = "error"
    else:
        status = "ok"

    return {
        "status": status,
        "service": "backend",
        "processor_mode": PROCESSOR_MODE,
        "ocr_mode": os.getenv("OCR_MODE", "qwen"),
        "model_name": os.getenv("MODEL_NAME", "qwen2.5vl:3b"),
        "model_base_url": os.getenv("MODEL_BASE_URL", "http://localhost:11434"),
        "yolo_model_path": os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt"),
        "pipeline_concurrency": PIPELINE_CONCURRENCY,
        "pipeline_device": PIPELINE_DEVICE,
        "pipeline_require_gpu": PIPELINE_REQUIRE_GPU,
        "yolo_device": YOLO_DEVICE,
        "easyocr_gpu": EASYOCR_GPU,
        "upload_limits": {
            "max_files": MAX_UPLOAD_FILES,
            "max_size_mb": MAX_UPLOAD_SIZE_MB,
            "allowed_extensions": sorted(ALLOWED_IMAGE_EXTENSIONS),
        },
        "dependencies": dependencies,
        "time": now_iso(),
    }


def check_jobs_dir() -> dict[str, Any]:
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        probe = JOBS_DIR / ".healthcheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"status": "ok", "path": str(JOBS_DIR)}
    except Exception as exc:
        return {
            "status": "error",
            "path": str(JOBS_DIR),
            "message": f"Jobs directory is not writable: {type(exc).__name__}: {exc}",
        }


def check_yolo_model() -> dict[str, Any]:
    yolo_path = Path(os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt"))
    if PROCESSOR_MODE == "mock":
        return {"status": "skipped", "reason": "mock processor mode"}
    if yolo_path.exists():
        return {"status": "ok", "path": str(yolo_path)}
    return {"status": "error", "path": str(yolo_path), "message": "YOLO model file not found"}


def check_backend_gpu() -> dict[str, Any]:
    if PROCESSOR_MODE == "mock":
        return {"status": "skipped", "reason": "mock processor mode"}

    requires_gpu = PIPELINE_REQUIRE_GPU or is_gpu_requested(PIPELINE_DEVICE, YOLO_DEVICE, EASYOCR_GPU)
    try:
        from data_extractor.runtime import torch_cuda_info
    except Exception as exc:  # pragma: no cover - environment-specific
        payload = {
            "status": "error" if requires_gpu else "skipped",
            "required": requires_gpu,
            "message": f"Cannot import data_extractor runtime: {type(exc).__name__}: {exc}",
        }
        return payload

    info = torch_cuda_info().to_dict()
    cuda_available = bool(info.get("cuda_available"))
    payload = {
        "status": "ok" if cuda_available else ("error" if requires_gpu else "skipped"),
        "required": requires_gpu,
        "pipeline_device": PIPELINE_DEVICE,
        "yolo_device": YOLO_DEVICE,
        "easyocr_gpu": EASYOCR_GPU,
        **info,
    }
    if not cuda_available:
        payload["message"] = "CUDA is not available inside backend container"
    return payload


def is_gpu_requested(*values: object) -> bool:
    for value in values:
        normalized = str(value or "").strip().lower()
        if normalized in {"1", "true", "yes", "on", "gpu", "cuda", "cuda:0", "0"}:
            return True
        if normalized.startswith("cuda"):
            return True
    return False


def check_ollama() -> dict[str, Any]:
    if PROCESSOR_MODE == "mock":
        return {"status": "skipped", "reason": "mock processor mode"}

    ocr_mode = os.getenv("OCR_MODE", "qwen").strip().lower()
    if ocr_mode != "qwen":
        return {"status": "skipped", "reason": f"ocr mode is {ocr_mode}"}

    base_url = os.getenv("MODEL_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = os.getenv("MODEL_NAME", "qwen2.5vl:3b")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "status": "error",
            "url": base_url,
            "model_name": model_name,
            "message": f"Ollama is not available: {type(exc).__name__}: {exc}",
        }

    models = payload.get("models") or []
    model_names = {str(item.get("name") or item.get("model") or "") for item in models if isinstance(item, dict)}
    if model_name in model_names:
        return {
            "status": "ok",
            "url": base_url,
            "model_name": model_name,
            "available_models": sorted(model_names),
        }

    return {
        "status": "error",
        "url": base_url,
        "model_name": model_name,
        "available_models": sorted(model_names),
        "message": "Required Ollama model is not pulled",
    }


def schedule_processing(job_id: str) -> None:
    task = _tasks.get(job_id)
    if task and not task.done():
        return
    _tasks[job_id] = asyncio.create_task(process_job(job_id))


async def process_job(job_id: str) -> None:
    async with _pipeline_semaphore:
        started = time.perf_counter()

        async with _store_lock:
            job = require_job(job_id)
            job["status"] = "running"
            job["status_text"] = "Обработка"
            job["message"] = "Файл обрабатывается"
            job["progress"] = 5
            job["attempt"] = int(job.get("attempt") or 0) + 1
            job["processor_mode"] = PROCESSOR_MODE
            job["error_code"] = None
            job["error_message"] = None
            save_job(job)
            job_snapshot = dict(job)

        try:
            if PROCESSOR_MODE == "mock":
                result = await run_mock_processor(job_id, job_snapshot)
            else:
                result = await run_pipeline_processor(job_id, job_snapshot)

            async with _store_lock:
                job = require_job(job_id)
                elapsed = time.perf_counter() - started
                apply_processor_result(job, result, elapsed)
                save_job(job)

        except Exception as exc:
            async with _store_lock:
                job = require_job(job_id)
                elapsed = time.perf_counter() - started
                error_text = f"{type(exc).__name__}: {exc}"
                status_text, public_message = friendly_error("WEB_PIPELINE_ERROR", error_text)
                job["status"] = "error"
                job["status_text"] = status_text
                job["student_number"] = None
                job["progress"] = 100
                job["message"] = public_message
                job["error_code"] = "WEB_PIPELINE_ERROR"
                job["error_message"] = error_text
                job["elapsed_sec"] = round(elapsed, 3)
                job["finished_at"] = now_iso()
                job["download_path"] = None
                job["result_path"] = None
                job["artifact_paths"] = {}
                job["artifacts"] = build_artifact_urls(job)
                save_job(job)


async def run_pipeline_processor(job_id: str, job_snapshot: dict[str, Any]) -> dict[str, Any]:
    task = asyncio.create_task(asyncio.to_thread(run_pipeline_sync, job_snapshot))
    progress_values = [12, 18, 24, 30, 36, 43, 50, 57, 64, 71, 78, 84, 89, 93, 96]
    progress_index = 0

    while not task.done():
        await asyncio.sleep(2.0)
        async with _store_lock:
            job = require_job(job_id)
            if job.get("status") != "running":
                continue
            if progress_index < len(progress_values):
                job["progress"] = progress_values[progress_index]
                progress_index += 1
            else:
                job["progress"] = min(97, int(job.get("progress") or 96))
            job["message"] = "Выполняется распознавание"
            save_job(job)

    return await task


def run_pipeline_sync(job: dict[str, Any]) -> dict[str, Any]:
    try:
        from data_extractor import DocumentPipeline, PipelineConfig
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("data_extractor is not installed or cannot be imported") from exc

    input_path = Path(job["input_path"])
    job_dir = JOBS_DIR / job["id"]
    output_dir = job_dir / "pipeline_out"
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt"),
        model_base_url=os.getenv("MODEL_BASE_URL", "http://localhost:11434"),
        model_name=os.getenv("MODEL_NAME", "qwen2.5vl:3b"),
        ocr_mode=os.getenv("OCR_MODE", "qwen"),
        mock_ocr_text=os.getenv("MOCK_OCR_TEXT", "№ 22-ЭТФ-062"),
        compute_device=os.getenv("PIPELINE_DEVICE", "auto"),
        yolo_device=None if os.getenv("YOLO_DEVICE", "auto").strip().lower() in {"", "auto", "default"} else os.getenv("YOLO_DEVICE"),
        easyocr_gpu=os.getenv("EASYOCR_GPU", "auto"),
    )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        pipeline = DocumentPipeline(config)
        pipeline_result = pipeline.process_image(str(input_path), str(output_dir), debug=PIPELINE_DEBUG)

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    if stdout_text.strip():
        (logs_dir / "engine_stdout.log").write_text(stdout_text, encoding="utf-8")
    if stderr_text.strip():
        (logs_dir / "engine_stderr.log").write_text(stderr_text, encoding="utf-8")

    result_dict = pipeline_result.to_dict()
    (job_dir / "pipeline_result.json").write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"kind": "pipeline", "result": result_dict, "output_dir": str(output_dir), "logs_dir": str(logs_dir)}


async def run_mock_processor(job_id: str, job_snapshot: dict[str, Any]) -> dict[str, Any]:
    for progress in [18, 32, 48, 64, 80, 94]:
        await asyncio.sleep(0.45)
        async with _store_lock:
            job = require_job(job_id)
            if job["status"] != "running":
                return {"kind": "mock", "cancelled": True}
            job["progress"] = progress
            save_job(job)

    await asyncio.sleep(0.35)
    failed = should_fail(job_snapshot)
    if failed:
        return {
            "kind": "mock",
            "status": "error",
            "student_number": None,
            "error_code": "NUMBER_NOT_RECOGNIZED",
            "error_message": "Номер не удалось определить",
        }

    number = guess_number(job_snapshot["filename"])
    return {
        "kind": "mock",
        "status": "ok",
        "student_number": number,
        "error_code": None,
        "error_message": None,
    }


def apply_processor_result(job: dict[str, Any], processor_result: dict[str, Any], elapsed: float) -> None:
    job["progress"] = 100
    job["elapsed_sec"] = round(elapsed, 3)
    job["finished_at"] = now_iso()

    if processor_result.get("kind") == "mock":
        status = processor_result.get("status")
        number = processor_result.get("student_number")
        job["pipeline_result"] = processor_result
        job["artifact_paths"] = {}
        job["result_path"] = job.get("input_path") if status == "ok" else None
        job["download_path"] = job.get("input_path") if status == "ok" else None

        if status == "ok" and number:
            job["status"] = "ok"
            job["status_text"] = "Готово"
            job["student_number"] = number
            job["message"] = "Номер распознан. Результат готов к скачиванию."
            job["error_code"] = None
            job["error_message"] = None
        else:
            error_code = processor_result.get("error_code") or "NUMBER_NOT_RECOGNIZED"
            error_message = processor_result.get("error_message")
            status_text, message = friendly_error(error_code, error_message)
            job["status"] = "error"
            job["status_text"] = status_text
            job["student_number"] = None
            job["message"] = message
            job["error_code"] = error_code
            job["error_message"] = error_message
            job["download_path"] = None
            job["result_path"] = None

        job["artifacts"] = build_artifact_urls(job)
        return

    result = processor_result.get("result") or {}
    status = result.get("status")
    number = result.get("student_number")

    face_path = result.get("face_path")
    anchor_path = result.get("anchor_path")
    recognized_anchor_path = result.get("recognized_anchor_path")
    yolo_debug_path = result.get("yolo_debug_path")

    artifact_paths = {
        "face": face_path,
        "result": face_path,  # финальный результат для пользователя — распознанное лицо
        "anchor": anchor_path,
        "recognized_anchor": recognized_anchor_path,
        "number_crop": recognized_anchor_path,
        "recognized": recognized_anchor_path,
        "yolo_debug": yolo_debug_path,
        "pipeline_result": str(JOBS_DIR / job["id"] / "pipeline_result.json"),
    }
    artifact_paths = {key: value for key, value in artifact_paths.items() if value}

    job["pipeline_result"] = result
    job["artifact_paths"] = artifact_paths
    job["result_path"] = artifact_paths.get("result")
    job["download_path"] = artifact_paths.get("result")

    if status == "ok" and number:
        job["status"] = "ok"
        job["status_text"] = "Готово"
        job["student_number"] = number
        job["message"] = "Номер распознан. Результат готов к скачиванию."
        job["error_code"] = None
        job["error_message"] = None
    else:
        error_code = result.get("error_code") or "NUMBER_NOT_RECOGNIZED"
        error_message = result.get("error_message")
        status_text, message = friendly_error(error_code, error_message)
        job["status"] = "error"
        job["status_text"] = status_text
        job["student_number"] = None
        job["message"] = message
        job["error_code"] = error_code
        job["error_message"] = error_message
        job["download_path"] = None
        job["result_path"] = None

    job["artifacts"] = build_artifact_urls(job)


def friendly_error(error_code: str | None, error_message: str | None = None) -> tuple[str, str]:
    """Return user-facing status and explanation for known pipeline errors."""
    code = (error_code or "").upper()
    message = (error_message or "").upper()
    source = f"{code} {message}"

    rules: list[tuple[tuple[str, ...], str, str]] = [
        (("NO_FACE", "FACE_NOT_FOUND", "NO_PERSON", "PERSON_NOT_FOUND", "NO_DETECTION", "PERSON"), "Лицо не найдено", "На изображении не удалось найти лицо. Проверьте, что лицо хорошо видно и не закрыто."),
        (("YOLO_MODEL", "YOLO_ERROR", "DETECTION_MODEL"), "Ошибка детекции", "Не удалось выполнить этап поиска лица. Повторите обработку или проверьте модель детекции."),
        (("ANCHOR_NOT_FOUND", "ANCHOR", "BOOK_NOT_FOUND", "GRADEBOOK_NOT_FOUND"), "Область номера не найдена", "Не удалось найти область с номером зачётной книжки."),
        (("TEXT_NOT_FOUND", "NO_TEXT", "OCR_EMPTY", "EMPTY_OCR"), "Текст не найден", "На найденной области не удалось прочитать текст."),
        (("OCR_DISABLED",), "OCR отключён", "Распознавание текста отключено в настройках сервера."),
        (("OCR_MODEL_ERROR", "OLLAMA", "HTTP 500", "MODEL_ERROR"), "Ошибка OCR-модели", "Модель распознавания не смогла обработать изображение. Повторите обработку после перезапуска модели."),
        (("NUMBER_NOT_RECOGNIZED", "PARSE_FAILED", "NUMBER"), "Номер не распознан", "Текст был обработан, но номер зачётной книжки определить не удалось."),
        (("JOB_INTERRUPTED",), "Обработка прервана", "Обработка была прервана. Запустите повторную обработку."),
        (("WEB_PIPELINE_ERROR",), "Ошибка обработки", "Во время обработки произошла ошибка. Подробности доступны в технических сведениях."),
    ]

    for markers, status_text, public_message in rules:
        if any(marker in source for marker in markers):
            return status_text, public_message

    return "Не распознано", "Номер не удалось определить. Попробуйте загрузить другое изображение или повторить обработку."


def should_fail(job: dict[str, Any]) -> bool:
    filename = job.get("filename") or ""
    attempt = int(job.get("attempt") or 1)
    return bool(re.search(r"failed|fail|bad|error|ошиб|плох", filename, re.IGNORECASE)) and attempt <= 1


def guess_number(filename: str) -> str:
    normalized = filename.upper()
    match = re.search(r"(\d{2})[-_ ]?(ЭТФ|ФПММ|ETF|FPMM)[-_ ]?(\d{3})", normalized)
    if match:
        faculty = match.group(2).replace("ETF", "ЭТФ").replace("FPMM", "ФПММ")
        return f"{match.group(1)}-{faculty}-{match.group(3)}"

    variants = ["22-ЭТФ-076", "22-ЭТФ-016", "22-ФПММ-241", "22-ЭТФ-047"]
    index = abs(hash(filename)) % len(variants)
    return variants[index]


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "filename": job.get("filename"),
        "size_bytes": job.get("size_bytes") or 0,
        "status": job.get("status"),
        "status_text": job.get("status_text"),
        "student_number": job.get("student_number"),
        "progress": job.get("progress") or 0,
        "message": job.get("message"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "elapsed_sec": job.get("elapsed_sec") or 0,
        "attempt": job.get("attempt") or 0,
        "processor_mode": job.get("processor_mode") or PROCESSOR_MODE,
        "error_code": job.get("error_code"),
        "error_message": job.get("error_message"),
        "artifacts": job.get("artifacts") or build_artifact_urls(job),
    }


def build_artifact_urls(job: dict[str, Any]) -> dict[str, str]:
    job_id = job["id"]
    return {
        "input": f"/api/jobs/{job_id}/artifacts/input",
        "source": f"/api/jobs/{job_id}/artifacts/input",
        "face": f"/api/jobs/{job_id}/artifacts/face",
        "anchor": f"/api/jobs/{job_id}/artifacts/anchor",
        "result": f"/api/jobs/{job_id}/artifacts/result",
        "recognized_anchor": f"/api/jobs/{job_id}/artifacts/recognized_anchor",
        "yolo_debug": f"/api/jobs/{job_id}/artifacts/yolo_debug",
    }


def resolve_artifact_path(job: dict[str, Any], artifact_name: str) -> Path | None:
    if artifact_name in {"input", "source", "original"}:
        return Path(job.get("input_path") or "")

    artifact_paths = job.get("artifact_paths") or {}
    alias_map = {
        "face": "face",
        "result": "result",
        "download": "result",
        "anchor": "anchor",
        "recognized": "recognized_anchor",
        "recognized_anchor": "recognized_anchor",
        "number_crop": "recognized_anchor",
        "yolo": "yolo_debug",
        "yolo_debug": "yolo_debug",
    }

    key = alias_map.get(artifact_name, artifact_name)
    value = artifact_paths.get(key)
    if value:
        return Path(value)

    if artifact_name in {"result", "download"} and job.get("result_path"):
        return Path(job["result_path"])

    return None


def require_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def load_jobs_from_disk() -> None:
    _jobs.clear()
    for job_file in JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not job.get("id"):
            continue

        if job.get("status") in {"queued", "running"}:
            job["status"] = "error"
            status_text, public_message = friendly_error("JOB_INTERRUPTED", "Server was restarted while the job was active.")
            job["status_text"] = status_text
            job["progress"] = 100
            job["message"] = public_message
            job["error_code"] = "JOB_INTERRUPTED"
            job["error_message"] = "Server was restarted while the job was active."
            job["finished_at"] = now_iso()
            save_job(job)

        _jobs[job["id"]] = job


def save_job(job: dict[str, Any]) -> None:
    job_dir = JOBS_DIR / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def get_upload_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return "jpg" if suffix == "jpeg" else suffix


def validate_upload(filename: str, extension: str, content_type: str | None, content: bytes) -> None:
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Недопустимый формат файла '{filename}'. Разрешены: {allowed}",
        )

    if not content:
        raise HTTPException(status_code=400, detail=f"Файл '{filename}' пустой")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Файл '{filename}' слишком большой. Максимум: {MAX_UPLOAD_SIZE_MB} МБ",
        )

    normalized_content_type = (content_type or "").lower().strip()
    if normalized_content_type and normalized_content_type != "application/octet-stream" and not normalized_content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Файл '{filename}' не похож на изображение: {normalized_content_type}",
        )

    if not looks_like_image(extension, content):
        raise HTTPException(
            status_code=415,
            detail=f"Файл '{filename}' не удалось определить как изображение",
        )


def looks_like_image(extension: str, content: bytes) -> bool:
    head = content[:32]
    if extension == "jpg":
        return head.startswith(b"\xff\xd8\xff")
    if extension == "png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if extension == "webp":
        return len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if extension == "bmp":
        return head.startswith(b"BM")
    return False


def get_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp", "bmp", "gif", "svg", "json"}:
        return "jpg" if suffix == "jpeg" else suffix
    return "jpg"


def media_type_for_extension(extension: str) -> str:
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "gif": "image/gif",
        "svg": "image/svg+xml; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }.get(extension.lower(), "application/octet-stream")


def unique_archive_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def safe_name(value: str) -> str:
    return re.sub(r"[^\wа-яА-ЯёЁ. -]+", "_", value).strip() or "file"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def svg_response(svg: str) -> Response:
    return Response(content=svg, media_type="image/svg+xml; charset=utf-8")


def make_svg_placeholder(title: str, job: dict[str, Any]) -> str:
    number = job.get("student_number") or ""
    subtitle = number or (job.get("filename") or "")
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">
  <rect width="1200" height="800" fill="#f7f8fb"/>
  <rect x="80" y="80" width="1040" height="640" rx="36" fill="#ffffff" stroke="#d7dce5" stroke-width="3"/>
  <text x="600" y="350" text-anchor="middle" font-family="Arial, sans-serif" font-size="46" font-weight="700" fill="#111827">{xml_escape(title)}</text>
  <text x="600" y="420" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" fill="#6b7280">{xml_escape(subtitle)}</text>
</svg>
""".strip()


def xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
