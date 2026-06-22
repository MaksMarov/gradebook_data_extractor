from __future__ import annotations

from typing import Any

import numpy as np

from data_extractor.config import PipelineConfig
from data_extractor.errors import FACE_DETECTION_ERROR, FACE_NOT_FOUND, PipelineStageError
from data_extractor.image.drawing import draw_bbox
from data_extractor.runtime import resolve_yolo_device
from data_extractor.schemas import FaceDetectionResult


class FaceDetector:
    """YOLO-based detector.

    The current model detects the `person` class. In the project domain this
    crop is used as the final face/person photo artifact. The name is kept as
    FaceDetector because this stage semantically produces the face/person crop.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model: Any | None = None
        self._device = resolve_yolo_device(config.compute_device, config.yolo_device)

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.config.yolo_model_path)
            except Exception as exc:
                raise PipelineStageError(
                    FACE_DETECTION_ERROR,
                    "YOLO model initialization failed",
                    {"model_path": self.config.yolo_model_path, "error": str(exc)},
                ) from exc
        return self._model

    def detect(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        try:
            predict_kwargs: dict[str, Any] = {
                "conf": self.config.yolo_conf_threshold,
                "verbose": False,
            }
            if self._device is not None:
                predict_kwargs["device"] = self._device
            results = self.model.predict(image_bgr, **predict_kwargs)
        except PipelineStageError:
            raise
        except Exception as exc:
            raise PipelineStageError(
                FACE_DETECTION_ERROR,
                "YOLO inference failed",
                {"error": str(exc), "device": self._device},
            ) from exc

        best_crop = None
        best_bbox: tuple[int, int, int, int] | None = None
        best_conf = 0.0
        best_label = ""

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                label = self.model.names[int(cls)]
                if label != self.config.yolo_target_label:
                    continue
                conf_float = float(conf)
                if conf_float < self.config.yolo_conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if conf_float > best_conf:
                    best_conf = conf_float
                    best_bbox = (x1, y1, x2, y2)
                    best_crop = crop
                    best_label = label

        if best_crop is None or best_bbox is None:
            raise PipelineStageError(
                FACE_NOT_FOUND,
                "No suitable person/face crop found",
                {"target_label": self.config.yolo_target_label, "conf_threshold": self.config.yolo_conf_threshold},
            )

        return FaceDetectionResult(
            found=True,
            bbox=best_bbox,
            confidence=round(best_conf, 4),
            label=best_label,
            face_image=best_crop,
            debug_image=draw_bbox(image_bgr, best_bbox, f"{best_label} {best_conf:.2f}"),
        )
