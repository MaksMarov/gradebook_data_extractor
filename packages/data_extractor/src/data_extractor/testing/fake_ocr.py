from __future__ import annotations

import numpy as np

from data_extractor.config import PipelineConfig
from data_extractor.schemas import OcrResult, VlmCall


class FakeOcrReader:
    """Deterministic OCR reader for tests and UI development without Ollama."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def recognize(self, anchor_image_bgr: np.ndarray, debug: bool = False) -> OcrResult:
        text = self.config.mock_ocr_text
        calls = [
            VlmCall("full", text, 0.0, ""),
            VlmCall("digits", " ".join([p for p in text.replace("-", " ").split() if any(ch.isdigit() for ch in p)]), 0.0, ""),
            VlmCall("faculty", "ЭТФ" if "ЭТФ" in text else "ФПММ" if "ФПММ" in text else "UNKNOWN", 0.0, ""),
        ]
        return OcrResult(
            status="ok",
            raw_full=calls[0].response,
            raw_digits=calls[1].response,
            raw_faculty=calls[2].response,
            calls=calls,
        )
