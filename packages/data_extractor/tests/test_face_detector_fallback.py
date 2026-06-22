from __future__ import annotations

import numpy as np
import pytest

from data_extractor.config import PipelineConfig
from data_extractor.errors import FACE_DETECTION_ERROR, PipelineStageError
from data_extractor.stages.face_detector import FaceDetector


class _FakeBoxes:
    xyxy = [[1, 2, 11, 12]]
    cls = [0]
    conf = [0.91]


class _FakeResult:
    boxes = _FakeBoxes()


class _GpuFailsCpuWorksModel:
    names = {0: "person"}

    def __init__(self) -> None:
        self.devices: list[str | None] = []

    def predict(self, image_bgr: np.ndarray, **kwargs):
        device = kwargs.get("device")
        self.devices.append(device)
        if device == "0":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return [_FakeResult()]


def test_yolo_gpu_error_falls_back_to_cpu() -> None:
    model = _GpuFailsCpuWorksModel()
    detector = FaceDetector(
        PipelineConfig(
            yolo_model_path="dummy.pt",
            compute_device="cuda",
            yolo_device="0",
            yolo_cpu_fallback_enabled=True,
        )
    )
    detector._model = model

    result = detector.detect(np.ones((20, 20, 3), dtype=np.uint8))

    assert result.found is True
    assert result.device == "cpu"
    assert "retried on CPU" in result.fallback_reason
    assert model.devices == ["0", "cpu"]


def test_yolo_gpu_error_can_fail_without_cpu_fallback() -> None:
    model = _GpuFailsCpuWorksModel()
    detector = FaceDetector(
        PipelineConfig(
            yolo_model_path="dummy.pt",
            compute_device="cuda",
            yolo_device="0",
            yolo_cpu_fallback_enabled=False,
        )
    )
    detector._model = model

    with pytest.raises(PipelineStageError) as exc_info:
        detector.detect(np.ones((20, 20, 3), dtype=np.uint8))

    assert exc_info.value.code == FACE_DETECTION_ERROR
    assert model.devices == ["0"]
