from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
import requests

from data_extractor.config import PipelineConfig
from data_extractor.errors import OCR_MODEL_ERROR, OCR_MODEL_TIMEOUT, PipelineStageError


class OllamaVisionClient:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.base_url = config.normalized_model_base_url()

    def check(self, timeout: int = 5) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            if not response.ok:
                return False, f"HTTP {response.status_code}: {response.text[:300]}"
            data = response.json()
            models = data.get("models", [])
            names = {item.get("name") for item in models if isinstance(item, dict)}
            if self.config.model_name and names and self.config.model_name not in names:
                return False, f"Model '{self.config.model_name}' not found. Available: {', '.join(sorted(n for n in names if n))}"
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def chat_with_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        image_bgr = self._prepare_image(image_bgr)
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise PipelineStageError(OCR_MODEL_ERROR, "Cannot encode anchor crop as JPEG")

        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "keep_alive": self.config.qwen_keep_alive,
            "options": {
                "temperature": self.config.qwen_temperature,
                "num_predict": self.config.qwen_num_predict,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.qwen_timeout_seconds,
            )
        except requests.Timeout as exc:
            raise PipelineStageError(OCR_MODEL_TIMEOUT, "Ollama request timed out", {"error": str(exc)}) from exc
        except Exception as exc:
            raise PipelineStageError(OCR_MODEL_ERROR, "Ollama request failed", {"error": str(exc)}) from exc

        if not response.ok:
            text = response.text
            code = OCR_MODEL_TIMEOUT if "timeout" in text.lower() else OCR_MODEL_ERROR
            raise PipelineStageError(code, f"Ollama HTTP {response.status_code}", {"response": text[:2000]})

        return str(response.json().get("message", {}).get("content", ""))

    def _prepare_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise PipelineStageError(OCR_MODEL_ERROR, "Empty image passed to Ollama client")
        h, w = image_bgr.shape[:2]
        max_width = self.config.qwen_max_image_width
        if max_width and w > max_width:
            scale = max_width / float(w)
            image_bgr = cv2.resize(image_bgr, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return image_bgr
