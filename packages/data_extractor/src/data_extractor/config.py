from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


def default_allowed_years() -> tuple[str, ...]:
    current = datetime.now().year % 100
    return tuple(f"{i:02d}" for i in range(current + 1))


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for one-image processing pipeline.

    The pipeline is intentionally independent from FastAPI, Docker and batch
    jobs. All runtime settings needed by the algorithm are represented here.
    """

    yolo_model_path: str

    model_base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5vl:3b"

    # qwen     - real Ollama/Qwen VLM OCR
    # mock     - deterministic fake OCR, useful for web/service tests
    # disabled - stop after face/anchor artifacts without calling OCR
    ocr_mode: Literal["qwen", "mock", "disabled"] = "qwen"
    mock_ocr_text: str = "№ 22-ЭТФ-062"

    allowed_faculties: tuple[str, ...] = ("ФПММ", "ЭТФ")
    allowed_years: tuple[str, ...] = field(default_factory=default_allowed_years)

    qwen_timeout_seconds: int = 300
    qwen_keep_alive: str = "0"
    qwen_temperature: float = 0.0
    qwen_num_predict: int = 80
    qwen_max_image_width: int = 1200

    # auto - use CUDA when torch can see it, otherwise CPU
    # cpu  - force CPU for YOLO/EasyOCR helper stages
    # cuda - force GPU and fail fast if CUDA is not available
    compute_device: str = "auto"
    yolo_device: str | None = None
    # When GPU inference fails (for example unsupported CUDA kernels on new GPUs),
    # retry YOLO on CPU instead of failing the whole pipeline.
    yolo_cpu_fallback_enabled: bool = True
    easyocr_gpu: bool | str = "auto"

    yolo_conf_threshold: float = 0.4
    yolo_target_label: str = "person"

    # Enlarged crop around the detected anchor text.
    # 3.6 keeps more context below the "Зачетная книжка" title for OCR.
    anchor_expand_ratio: float = 3.6
    image_max_side: int | None = None
    save_debug: bool = True

    def normalized_model_base_url(self) -> str:
        return self.model_base_url.rstrip("/")
