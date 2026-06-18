from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from data_extractor.config import PipelineConfig
from data_extractor.errors import (
    NUMBER_NOT_RECOGNIZED,
    OCR_DISABLED,
    PIPELINE_UNHANDLED_ERROR,
    PipelineStageError,
)
from data_extractor.schemas import PipelineResult
from data_extractor.stages.anchor_finder import AnchorFinder
from data_extractor.stages.anchor_ocr import AnchorOcrReader
from data_extractor.stages.artifact_writer import ArtifactWriter
from data_extractor.stages.face_detector import FaceDetector
from data_extractor.stages.image_loader import ImageLoader
from data_extractor.stages.number_parser import StudentNumberParser
from data_extractor.stages.qwen_ocr import DisabledOcrReader, QwenOcrReader
from data_extractor.testing.fake_ocr import FakeOcrReader
from data_extractor.utils.timing import timed


class DocumentPipeline:
    """One image -> one structured PipelineResult.

    This class is the public API of the package. It does not depend on FastAPI,
    Docker, web UI, batch jobs, or service storage.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.image_loader = ImageLoader(config)
        self.face_detector = FaceDetector(config)
        self.anchor_ocr = AnchorOcrReader(config)
        self.anchor_finder = AnchorFinder(config)
        self.ocr_reader = self._build_ocr_reader(config)
        self.number_parser = StudentNumberParser(config)
        self.artifacts = ArtifactWriter(config)

    def process_image(self, image_path: str | Path, output_dir: str | Path, debug: bool | None = None) -> PipelineResult:
        output_dir = self.artifacts.prepare_output_dir(output_dir)
        debug_enabled = self.config.save_debug if debug is None else debug
        timings: dict[str, float] = {}
        debug_data: dict[str, Any] = {
            "source_path": str(image_path),
            "output_dir": str(output_dir),
            "config": self._config_debug_dict(),
            "stages": {},
        }

        face_path: str | None = None
        anchor_path: str | None = None
        recognized_anchor_path: str | None = None
        yolo_debug_path: str | None = None

        try:
            if debug_enabled:
                with timed(timings, "copy_source_ref"):
                    debug_data["source_ref_path"] = self.artifacts.copy_source_ref(image_path, output_dir)

            with timed(timings, "load_image"):
                image_data = self.image_loader.load(image_path)
            debug_data["stages"]["image_loader"] = {
                "width": image_data.width,
                "height": image_data.height,
            }

            with timed(timings, "face_detection"):
                face_result = self.face_detector.detect(image_data.image_bgr)
            debug_data["stages"]["face_detection"] = face_result.to_debug_dict()

            with timed(timings, "save_face"):
                face_path = self.artifacts.save_face(face_result.face_image, output_dir)
                if debug_enabled and face_result.debug_image is not None:
                    yolo_debug_path = self.artifacts.save_yolo_debug(face_result.debug_image, output_dir)

            with timed(timings, "anchor_ocr"):
                anchor_ocr_items = self.anchor_ocr.read(image_data.image_bgr)
            debug_data["stages"]["anchor_ocr"] = {
                "items": [item.to_dict() for item in anchor_ocr_items],
            }

            with timed(timings, "anchor_detection"):
                anchor_result = self.anchor_finder.find(image_data.image_bgr, anchor_ocr_items)
            debug_data["stages"]["anchor_detection"] = anchor_result.to_debug_dict()

            with timed(timings, "save_anchor"):
                anchor_path = self.artifacts.save_anchor(anchor_result.anchor_image, output_dir)

            with timed(timings, "ocr"):
                ocr_result = self.ocr_reader.recognize(anchor_result.anchor_image, debug=debug_enabled)
            debug_data["stages"]["ocr"] = ocr_result.to_dict()

            if ocr_result.status == "disabled":
                result = PipelineResult.error(
                    code=OCR_DISABLED,
                    message=ocr_result.error_message or "OCR disabled",
                    source_path=str(image_path),
                    output_dir=str(output_dir),
                    timings=timings,
                    debug=debug_data,
                    face_path=face_path,
                    anchor_path=anchor_path,
                    yolo_debug_path=yolo_debug_path,
                )
                self._save_final_artifacts(result, output_dir, debug_data)
                return result

            with timed(timings, "number_parse"):
                number_result = self.number_parser.parse_ocr(ocr_result)
            debug_data["stages"]["number_parse"] = number_result.to_dict()

            if not number_result.success:
                recognized_anchor_path = self.artifacts.save_recognized_anchor(anchor_result.anchor_image, output_dir, None)
                result = PipelineResult.error(
                    code=NUMBER_NOT_RECOGNIZED,
                    message=number_result.error or "Student number was not recognized",
                    source_path=str(image_path),
                    output_dir=str(output_dir),
                    timings=timings,
                    debug=debug_data,
                    face_path=face_path,
                    anchor_path=anchor_path,
                    recognized_anchor_path=recognized_anchor_path,
                    yolo_debug_path=yolo_debug_path,
                )
                result.raw_full = ocr_result.raw_full
                result.raw_digits = ocr_result.raw_digits
                result.raw_faculty = ocr_result.raw_faculty
                result.parser_note = number_result.note
                result.recognition_status = number_result.recognition_status
                self._save_final_artifacts(result, output_dir, debug_data)
                return result

            recognized_anchor_path = self.artifacts.save_recognized_anchor(
                anchor_result.anchor_image,
                output_dir,
                number_result.student_number,
            )

            result = PipelineResult(
                status="ok",
                error_code=None,
                error_message=None,
                student_number=number_result.student_number,
                source_path=str(image_path),
                output_dir=str(output_dir),
                face_path=face_path,
                anchor_path=anchor_path,
                recognized_anchor_path=recognized_anchor_path,
                yolo_debug_path=yolo_debug_path,
                recognition_status=number_result.recognition_status,
                year=number_result.year,
                faculty=number_result.faculty,
                serial=number_result.serial,
                year_score=number_result.year_score,
                faculty_score=number_result.faculty_score,
                serial_score=number_result.serial_score,
                raw_full=ocr_result.raw_full,
                raw_digits=ocr_result.raw_digits,
                raw_faculty=ocr_result.raw_faculty,
                parser_note=number_result.note,
                timings=timings,
                debug=debug_data,
            )
            self._save_final_artifacts(result, output_dir, debug_data)
            return result

        except PipelineStageError as exc:
            debug_data.setdefault("exception", {})
            debug_data["exception"] = {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
            result = PipelineResult.error(
                code=exc.code,
                message=exc.message,
                source_path=str(image_path),
                output_dir=str(output_dir),
                timings=timings,
                debug=debug_data,
                face_path=face_path,
                anchor_path=anchor_path,
                recognized_anchor_path=recognized_anchor_path,
                yolo_debug_path=yolo_debug_path,
            )
            self._save_final_artifacts_safely(result, output_dir, debug_data)
            return result
        except Exception as exc:
            debug_data["exception"] = {
                "code": PIPELINE_UNHANDLED_ERROR,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            result = PipelineResult.error(
                code=PIPELINE_UNHANDLED_ERROR,
                message="Unhandled pipeline error",
                source_path=str(image_path),
                output_dir=str(output_dir),
                timings=timings,
                debug=debug_data,
                face_path=face_path,
                anchor_path=anchor_path,
                recognized_anchor_path=recognized_anchor_path,
                yolo_debug_path=yolo_debug_path,
            )
            self._save_final_artifacts_safely(result, output_dir, debug_data)
            return result
        finally:
            if timings:
                timings["total_recorded"] = round(sum(timings.values()), 4)

    def process_image_dict(self, image_path: str | Path, output_dir: str | Path, debug: bool | None = None) -> dict[str, Any]:
        return self.process_image(image_path, output_dir, debug=debug).to_dict()

    def _save_final_artifacts(self, result: PipelineResult, output_dir: Path, debug_data: dict[str, Any]) -> None:
        debug_data["result_status"] = result.status
        debug_data["result_error_code"] = result.error_code
        self.artifacts.save_debug_json(debug_data, output_dir)
        self.artifacts.save_result_json(result, output_dir)

    def _save_final_artifacts_safely(self, result: PipelineResult, output_dir: Path, debug_data: dict[str, Any]) -> None:
        try:
            self._save_final_artifacts(result, output_dir, debug_data)
        except Exception:
            # Avoid hiding original processing error behind debug-save error.
            pass

    def _build_ocr_reader(self, config: PipelineConfig):
        if config.ocr_mode == "disabled":
            return DisabledOcrReader(config)
        if config.ocr_mode == "mock":
            return FakeOcrReader(config)
        if config.ocr_mode == "qwen":
            return QwenOcrReader(config)
        raise ValueError(f"Unsupported OCR mode: {config.ocr_mode}")

    def _config_debug_dict(self) -> dict[str, Any]:
        return {
            "yolo_model_path": self.config.yolo_model_path,
            "model_base_url": self.config.model_base_url,
            "model_name": self.config.model_name,
            "ocr_mode": self.config.ocr_mode,
            "allowed_faculties": list(self.config.allowed_faculties),
            "allowed_years": list(self.config.allowed_years),
            "anchor_expand_ratio": self.config.anchor_expand_ratio,
            "image_max_side": self.config.image_max_side,
        }


class StudDocPipeline:
    """Compatibility adapter for the old project API.

    Old code used:
        pipeline = StudDocPipeline(...)
        result: dict = pipeline.process_image(...)

    New code should use DocumentPipeline + PipelineConfig and receive a
    PipelineResult object.
    """

    def __init__(
        self,
        yolo_model_path: str,
        qwen_model: str = "qwen2.5vl:3b",
        qwen_base_url: str = "http://localhost:11434",
        qwen_years: list[str] | None = None,
        qwen_timeout: int = 300,
        qwen_keep_alive: str = "0",
        qwen_num_predict: int = 80,
        qwen_temperature: float = 0.0,
        qwen_max_image_width: int = 1200,
        anchor_expand_ratio: float = 3.0,
        ocr_mode: str = "qwen",
    ):
        config = PipelineConfig(
            yolo_model_path=yolo_model_path,
            model_name=qwen_model,
            model_base_url=qwen_base_url,
            allowed_years=tuple(qwen_years) if qwen_years else PipelineConfig(yolo_model_path=yolo_model_path).allowed_years,
            qwen_timeout_seconds=qwen_timeout,
            qwen_keep_alive=qwen_keep_alive,
            qwen_num_predict=qwen_num_predict,
            qwen_temperature=qwen_temperature,
            qwen_max_image_width=qwen_max_image_width,
            anchor_expand_ratio=anchor_expand_ratio,
            ocr_mode=ocr_mode,  # type: ignore[arg-type]
        )
        self._pipeline = DocumentPipeline(config)

    def process_image(self, img_path: str, output_dir: str, debug: bool = False, debug_path: str | None = None) -> dict[str, Any]:
        result = self._pipeline.process_image(img_path, output_dir, debug=debug).to_dict()
        # Compatibility keys used by current service layer.
        result["status"] = "ok" if result.get("status") == "ok" else "error"
        result["reason"] = result.get("error_message")
        result["student_number"] = result.get("student_number") or ""
        result["qwen_calls"] = result.get("debug", {}).get("stages", {}).get("ocr", {}).get("calls", [])
        result["anchor_found"] = bool(result.get("anchor_path"))
        return result
