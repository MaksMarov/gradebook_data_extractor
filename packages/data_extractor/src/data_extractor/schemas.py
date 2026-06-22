from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from data_extractor.utils.json_utils import to_jsonable


@dataclass(slots=True)
class ImageData:
    source_path: str
    image_bgr: Any
    width: int
    height: int


@dataclass(slots=True)
class FaceDetectionResult:
    found: bool
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    label: str = ""
    face_image: Any = None
    debug_image: Any = None
    error: str = ""
    device: str = ""
    fallback_reason: str = ""

    def to_debug_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "found": self.found,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "label": self.label,
            "error": self.error,
            "device": self.device,
            "fallback_reason": self.fallback_reason,
        })


@dataclass(slots=True)
class AnchorOcrItem:
    bbox: list[list[float]]
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class AnchorResult:
    found: bool
    bbox: list[list[float]] | None = None
    anchor_image: Any = None
    matched_text: str = ""
    confidence: float = 0.0
    error: str = ""

    def to_debug_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "found": self.found,
            "bbox": self.bbox,
            "matched_text": self.matched_text,
            "confidence": self.confidence,
            "error": self.error,
        })


@dataclass(slots=True)
class VlmCall:
    call_name: str
    response: str
    elapsed_sec: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class OcrResult:
    status: Literal["ok", "error", "disabled"]
    raw_full: str = ""
    raw_digits: str = ""
    raw_faculty: str = ""
    calls: list[VlmCall] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "raw_full": self.raw_full,
            "raw_digits": self.raw_digits,
            "raw_faculty": self.raw_faculty,
            "calls": [c.to_dict() for c in self.calls],
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class NumberParseResult:
    success: bool
    student_number: str = ""
    year: str = ""
    faculty: str = ""
    serial: str = ""
    year_score: float = 0.0
    faculty_score: float = 0.0
    serial_score: float = 0.0
    recognition_status: str = ""
    note: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class PipelineResult:
    status: Literal["ok", "error"]
    error_code: str | None
    error_message: str | None

    student_number: str | None
    source_path: str
    output_dir: str

    face_path: str | None = None
    anchor_path: str | None = None
    recognized_anchor_path: str | None = None
    yolo_debug_path: str | None = None

    recognition_status: str | None = None
    year: str | None = None
    faculty: str | None = None
    serial: str | None = None
    year_score: float = 0.0
    faculty_score: float = 0.0
    serial_score: float = 0.0

    raw_full: str | None = None
    raw_digits: str | None = None
    raw_faculty: str | None = None
    parser_note: str | None = None

    timings: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def error(
        cls,
        *,
        code: str,
        message: str,
        source_path: str,
        output_dir: str,
        timings: dict[str, float] | None = None,
        debug: dict[str, Any] | None = None,
        face_path: str | None = None,
        anchor_path: str | None = None,
        recognized_anchor_path: str | None = None,
        yolo_debug_path: str | None = None,
    ) -> "PipelineResult":
        return cls(
            status="error",
            error_code=code,
            error_message=message,
            student_number=None,
            source_path=source_path,
            output_dir=output_dir,
            face_path=face_path,
            anchor_path=anchor_path,
            recognized_anchor_path=recognized_anchor_path,
            yolo_debug_path=yolo_debug_path,
            timings=timings or {},
            debug=debug or {},
        )
