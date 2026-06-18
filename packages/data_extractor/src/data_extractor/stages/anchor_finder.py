from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from data_extractor.config import PipelineConfig
from data_extractor.errors import ANCHOR_DETECTION_ERROR, ANCHOR_NOT_FOUND, PipelineStageError
from data_extractor.image.drawing import polygon_to_rect
from data_extractor.schemas import AnchorOcrItem, AnchorResult


class AnchorFinder:
    def __init__(self, config: PipelineConfig, keywords: tuple[str, ...] | None = None):
        self.config = config
        self.keywords = keywords or ("ЗАЧЕТНАЯ КНИЖКА", "СТУДЕНЧЕСКИЙ БИЛЕТ")

    def find(self, image_bgr: np.ndarray, ocr_items: list[AnchorOcrItem]) -> AnchorResult:
        for item in ocr_items:
            text_u = item.text.upper()
            for keyword in self.keywords:
                if keyword in text_u or SequenceMatcher(None, text_u, keyword).ratio() > 0.6:
                    anchor_image = self.crop_expanded(image_bgr, item.bbox, self.config.anchor_expand_ratio)
                    return AnchorResult(
                        found=True,
                        bbox=item.bbox,
                        anchor_image=anchor_image,
                        matched_text=item.text,
                        confidence=item.confidence,
                    )

        raise PipelineStageError(
            ANCHOR_NOT_FOUND,
            "Anchor text was not found",
            {"keywords": list(self.keywords), "ocr_texts": [item.text for item in ocr_items[:30]]},
        )

    @staticmethod
    def crop_expanded(image_bgr: np.ndarray, bbox: list[list[float]], ratio: float) -> np.ndarray:
        try:
            x1, y1, x2, y2 = polygon_to_rect(bbox)
            w = x2 - x1
            h = y2 - y1
            cx = int(x1 + w / 2)
            cy = int(y1 + h / 2)
            nw = int(w * ratio)
            nh = int(h * ratio)
            nx1 = int(max(cx - nw // 2, 0))
            ny1 = int(max(cy - nh // 2, 0))
            nx2 = int(min(cx + nw // 2, image_bgr.shape[1]))
            ny2 = int(min(cy + nh // 2, image_bgr.shape[0]))
            crop = image_bgr[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                raise ValueError("Expanded anchor crop is empty")
            return crop
        except Exception as exc:
            raise PipelineStageError(ANCHOR_DETECTION_ERROR, "Failed to crop expanded anchor region", {"error": str(exc)}) from exc
