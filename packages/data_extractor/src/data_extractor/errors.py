from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


IMAGE_LOAD_ERROR = "IMAGE_LOAD_ERROR"
FACE_DETECTION_ERROR = "FACE_DETECTION_ERROR"
FACE_NOT_FOUND = "FACE_NOT_FOUND"

ANCHOR_OCR_ERROR = "ANCHOR_OCR_ERROR"
ANCHOR_DETECTION_ERROR = "ANCHOR_DETECTION_ERROR"
ANCHOR_NOT_FOUND = "ANCHOR_NOT_FOUND"

OCR_DISABLED = "OCR_DISABLED"
OCR_MODEL_UNAVAILABLE = "OCR_MODEL_UNAVAILABLE"
OCR_MODEL_TIMEOUT = "OCR_MODEL_TIMEOUT"
OCR_MODEL_ERROR = "OCR_MODEL_ERROR"
OCR_EMPTY_RESPONSE = "OCR_EMPTY_RESPONSE"

NUMBER_PARSE_ERROR = "NUMBER_PARSE_ERROR"
NUMBER_NOT_RECOGNIZED = "NUMBER_NOT_RECOGNIZED"

ARTIFACT_SAVE_ERROR = "ARTIFACT_SAVE_ERROR"
PIPELINE_UNHANDLED_ERROR = "PIPELINE_UNHANDLED_ERROR"


@dataclass(slots=True)
class PipelineStageError(Exception):
    """Typed internal exception used inside stages.

    The public API should return PipelineResult, not leak random exceptions.
    This exception is useful inside the orchestrator to convert failures into
    stable error_code/error_message values.
    """

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def error_code(self) -> str:
        return self.code

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
