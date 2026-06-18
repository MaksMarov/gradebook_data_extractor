from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from data_extractor import DocumentPipeline, PipelineConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class ConsoleFileResult:
    source_path: str
    source_filename: str
    output_dir: str
    status: str
    error_code: str | None
    error_message: str | None
    student_number: str | None
    expected_number: str | None = None
    exact_ok: bool | None = None
    year_ok: bool | None = None
    faculty_ok: bool | None = None
    serial_ok: bool | None = None
    face_path: str | None = None
    anchor_path: str | None = None
    recognized_anchor_path: str | None = None
    yolo_debug_path: str | None = None
    elapsed_sec: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_extractor.py",
        description="Console application for GradebookDataExtractor.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_common_args(
        subparsers.add_parser(
            "single",
            help="Process one image.",
        )
    ).add_argument(
        "--image",
        required=True,
        help="Path to image file.",
    )

    folder_parser = add_common_args(
        subparsers.add_parser(
            "folder",
            help="Process all images in a folder.",
        )
    )
    folder_parser.add_argument(
        "--input",
        required=True,
        help="Input folder with images.",
    )
    folder_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images recursively.",
    )
    folder_parser.add_argument(
        "--expected",
        default=None,
        help="Optional CSV file with columns: filename,expected_number.",
    )
    folder_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )

    return parser


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--yolo-model",
        default=os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt"),
        help="Path to YOLO model.",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=["disabled", "mock", "qwen"],
        default=os.getenv("OCR_MODE", "disabled"),
        help="OCR mode.",
    )
    parser.add_argument(
        "--model-url",
        default=os.getenv("MODEL_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "qwen2.5vl:3b"),
        help="Ollama model name.",
    )
    parser.add_argument(
        "--mock-ocr-text",
        default=os.getenv("MOCK_OCR_TEXT", "№ 22-ЭТФ-062"),
        help="Text used in mock OCR mode.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug artifacts.",
    )
    parser.add_argument(
        "--show-engine-output",
        action="store_true",
        help="Do not suppress verbose output from YOLO/EasyOCR/Qwen.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop folder processing after first error.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "single":
            return run_single_command(args)
        if args.command == "folder":
            return run_folder_command(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 1


def run_single_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = create_pipeline(args)

    result = process_one_image(
        pipeline=pipeline,
        image_path=Path(args.image),
        output_dir=output_dir,
        debug=args.debug,
        show_engine_output=args.show_engine_output,
    )

    print_single_result(result)

    write_json(output_dir / "console_result.json", asdict(result))
    write_csv(output_dir / "console_result.csv", [result])

    return 0 if result.status == "ok" else 2


def run_folder_command(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    images = list(find_images(input_dir, recursive=args.recursive))
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print(f"No images found in: {input_dir}", file=sys.stderr)
        return 1

    expected_map = load_expected_csv(Path(args.expected)) if args.expected else {}

    pipeline = create_pipeline(args)

    print("GradebookDataExtractor console")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_root}")
    print(f"Files:  {len(images)}")
    print(f"OCR:    {args.ocr_mode}")
    if args.ocr_mode == "qwen":
        print(f"Model:  {args.model_name} @ {args.model_url}")
    print("")

    results: list[ConsoleFileResult] = []
    started = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        rel_name = image_path.relative_to(input_dir).as_posix()
        item_output_dir = output_root / make_output_dir_name(index, image_path)

        print(f"[{index}/{len(images)}] {rel_name} ... ", end="", flush=True)

        result = process_one_image(
            pipeline=pipeline,
            image_path=image_path,
            output_dir=item_output_dir,
            debug=args.debug,
            show_engine_output=args.show_engine_output,
        )

        expected_number = expected_map.get(image_path.name) or expected_map.get(rel_name)
        if expected_number:
            apply_expected(result, expected_number)

        results.append(result)

        status_text = format_status(result)
        print(status_text)

        write_json(output_root / "results.json", [asdict(r) for r in results])
        write_csv(output_root / "results.csv", results)
        write_json(output_root / "summary.json", build_summary(results, time.perf_counter() - started))

        if args.fail_fast and result.status != "ok":
            break

    summary = build_summary(results, time.perf_counter() - started)
    print_summary(summary)

    return 0 if summary["failed"] == 0 else 2


def create_pipeline(args: argparse.Namespace) -> DocumentPipeline:
    config = PipelineConfig(
        yolo_model_path=args.yolo_model,
        model_base_url=args.model_url,
        model_name=args.model_name,
        ocr_mode=args.ocr_mode,
        mock_ocr_text=args.mock_ocr_text,
    )
    return DocumentPipeline(config)


def process_one_image(
    *,
    pipeline: DocumentPipeline,
    image_path: Path,
    output_dir: Path,
    debug: bool,
    show_engine_output: bool,
) -> ConsoleFileResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if show_engine_output:
            pipeline_result = pipeline.process_image(
                str(image_path),
                str(output_dir),
                debug=debug,
            )
        else:
            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                pipeline_result = pipeline.process_image(
                    str(image_path),
                    str(output_dir),
                    debug=debug,
                )
            save_engine_output(output_dir, captured_stdout.getvalue(), captured_stderr.getvalue())

        result_dict = pipeline_result.to_dict()
        return ConsoleFileResult(
            source_path=str(image_path),
            source_filename=image_path.name,
            output_dir=str(output_dir),
            status=str(result_dict.get("status") or "error"),
            error_code=result_dict.get("error_code"),
            error_message=result_dict.get("error_message"),
            student_number=result_dict.get("student_number"),
            face_path=result_dict.get("face_path"),
            anchor_path=result_dict.get("anchor_path"),
            recognized_anchor_path=result_dict.get("recognized_anchor_path"),
            yolo_debug_path=result_dict.get("yolo_debug_path"),
            elapsed_sec=round(time.perf_counter() - started, 3),
        )

    except Exception as exc:
        return ConsoleFileResult(
            source_path=str(image_path),
            source_filename=image_path.name,
            output_dir=str(output_dir),
            status="error",
            error_code="CONSOLE_UNHANDLED_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
            student_number=None,
            elapsed_sec=round(time.perf_counter() - started, 3),
        )


def find_images(input_dir: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def make_output_dir_name(index: int, image_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", image_path.stem).strip("_")
    return f"{index:04d}_{safe_stem}"


def format_status(result: ConsoleFileResult) -> str:
    elapsed = f"{result.elapsed_sec:.1f}s"

    if result.status == "ok":
        expected_suffix = ""
        if result.expected_number:
            expected_suffix = " expected=OK" if result.exact_ok else f" expected={result.expected_number}"
        return f"OK {result.student_number} ({elapsed}){expected_suffix}"

    code = result.error_code or "ERROR"
    message = result.error_message or ""
    if len(message) > 120:
        message = message[:117] + "..."
    return f"FAILED {code} ({elapsed}) {message}".rstrip()


def print_single_result(result: ConsoleFileResult) -> None:
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


def print_summary(summary: dict[str, Any]) -> None:
    print("")
    print("Summary")
    print(f"  total:   {summary['total']}")
    print(f"  success: {summary['success']}")
    print(f"  failed:  {summary['failed']}")
    print(f"  elapsed: {summary['elapsed_sec']}s")

    metrics = summary.get("metrics")
    if metrics:
        print("  metrics:")
        print(f"    exact:   {metrics['exact']['ok']}/{metrics['exact']['total']} = {metrics['exact']['percent']}%")
        print(f"    year:    {metrics['year']['ok']}/{metrics['year']['total']} = {metrics['year']['percent']}%")
        print(f"    faculty: {metrics['faculty']['ok']}/{metrics['faculty']['total']} = {metrics['faculty']['percent']}%")
        print(f"    serial:  {metrics['serial']['ok']}/{metrics['serial']['total']} = {metrics['serial']['percent']}%")


def build_summary(results: list[ConsoleFileResult], elapsed_sec: float) -> dict[str, Any]:
    total = len(results)
    success = sum(1 for r in results if r.status == "ok")
    failed = total - success

    summary: dict[str, Any] = {
        "total": total,
        "success": success,
        "failed": failed,
        "elapsed_sec": round(elapsed_sec, 3),
    }

    expected_results = [r for r in results if r.expected_number]
    if expected_results:
        summary["metrics"] = {
            "exact": metric(expected_results, "exact_ok"),
            "year": metric(expected_results, "year_ok"),
            "faculty": metric(expected_results, "faculty_ok"),
            "serial": metric(expected_results, "serial_ok"),
        }

    return summary


def metric(results: list[ConsoleFileResult], attr: str) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if getattr(r, attr) is True)
    percent = round(ok / total * 100, 2) if total else 0.0
    return {"ok": ok, "total": total, "percent": percent}


def load_expected_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"filename", "expected_number"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Expected CSV must contain columns: filename,expected_number")

        result: dict[str, str] = {}
        for row in reader:
            filename = (row.get("filename") or "").strip()
            expected_number = (row.get("expected_number") or "").strip()
            if filename and expected_number:
                result[filename] = normalize_number(expected_number)
        return result


def apply_expected(result: ConsoleFileResult, expected_number: str) -> None:
    expected = normalize_number(expected_number)
    actual = normalize_number(result.student_number or "")

    result.expected_number = expected
    result.exact_ok = actual == expected

    actual_parts = split_number(actual)
    expected_parts = split_number(expected)

    result.year_ok = actual_parts.get("year") == expected_parts.get("year")
    result.faculty_ok = actual_parts.get("faculty") == expected_parts.get("faculty")
    result.serial_ok = actual_parts.get("serial") == expected_parts.get("serial")


def normalize_number(value: str) -> str:
    value = value.strip().upper().replace(" ", "")
    value = value.replace("–", "-").replace("—", "-").replace("_", "-")
    return value


def split_number(value: str) -> dict[str, str]:
    match = re.search(r"(?P<year>\d{2})-(?P<faculty>[А-ЯA-Z]+)-(?P<serial>\d{3})", value)
    if not match:
        return {"year": "", "faculty": "", "serial": ""}
    return match.groupdict()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, results: list[ConsoleFileResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(ConsoleFileResult("", "", "", "", None, None, None)).keys())

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_engine_output(output_dir: Path, stdout_text: str, stderr_text: str) -> None:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if stdout_text.strip():
        (logs_dir / "engine_stdout.log").write_text(stdout_text, encoding="utf-8")
    if stderr_text.strip():
        (logs_dir / "engine_stderr.log").write_text(stderr_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
