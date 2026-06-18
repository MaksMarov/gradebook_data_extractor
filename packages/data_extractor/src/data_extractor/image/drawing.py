from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def draw_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], label: str = "") -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(out, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def polygon_to_rect(poly: Iterable[Iterable[float]]) -> tuple[int, int, int, int]:
    pts = [(float(x), float(y)) for x, y in poly]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
