from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def to_gray(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def increase_contrast(crop: np.ndarray) -> np.ndarray:
    gray = to_gray(crop)
    return cv2.equalizeHist(gray)


def adaptive_thresh(crop: np.ndarray) -> np.ndarray:
    gray = to_gray(crop)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def resize(crop: np.ndarray, fx: float = 2.0, fy: float = 2.0) -> np.ndarray:
    return cv2.resize(crop, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def limit_max_side(image: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return image
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image
    scale = max_side / float(current_max)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
