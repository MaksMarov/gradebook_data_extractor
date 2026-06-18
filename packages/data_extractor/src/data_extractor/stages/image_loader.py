from __future__ import annotations

from pathlib import Path

import cv2

from data_extractor.config import PipelineConfig
from data_extractor.errors import IMAGE_LOAD_ERROR, PipelineStageError
from data_extractor.image.preprocess import limit_max_side
from data_extractor.schemas import ImageData


class ImageLoader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load(self, image_path: str | Path) -> ImageData:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise PipelineStageError(IMAGE_LOAD_ERROR, "Image file does not exist", {"path": str(path)})

        image = cv2.imread(str(path))
        if image is None:
            raise PipelineStageError(IMAGE_LOAD_ERROR, "OpenCV could not load image", {"path": str(path)})

        image = limit_max_side(image, self.config.image_max_side)
        h, w = image.shape[:2]
        return ImageData(source_path=str(path), image_bgr=image, width=w, height=h)
