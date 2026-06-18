from __future__ import annotations

import re
from typing import Any

import numpy as np

from data_extractor.config import PipelineConfig
from data_extractor.errors import ANCHOR_OCR_ERROR, PipelineStageError
from data_extractor.schemas import AnchorOcrItem


class AnchorOcrReader:
    """EasyOCR wrapper used only to locate the printed anchor text."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._reader: Any | None = None

    @property
    def reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(["ru"], gpu=self.config.easyocr_gpu)
            except Exception as exc:
                raise PipelineStageError(ANCHOR_OCR_ERROR, "EasyOCR reader initialization failed", {"error": str(exc)}) from exc
        return self._reader

    def read(self, image_bgr: np.ndarray) -> list[AnchorOcrItem]:
        try:
            raw_results = self.reader.readtext(image_bgr)
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError(ANCHOR_OCR_ERROR, "EasyOCR anchor read failed", {"error": str(exc)}) from exc

        items: list[AnchorOcrItem] = []
        for bbox, text, conf in raw_results:
            normalized = self._postprocess_anchor_text(text)
            items.append(AnchorOcrItem(bbox=bbox, text=normalized, confidence=float(conf)))
        return items

    @staticmethod
    def _postprocess_anchor_text(text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"[^А-Яа-яA-Za-z ]", "", text)
        return text.upper()
