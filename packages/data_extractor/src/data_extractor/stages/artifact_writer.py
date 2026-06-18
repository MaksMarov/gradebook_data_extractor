from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_extractor.config import PipelineConfig
from data_extractor.errors import ARTIFACT_SAVE_ERROR, PipelineStageError
from data_extractor.schemas import PipelineResult
from data_extractor.utils.json_utils import write_json
from data_extractor.utils.paths import ensure_dir, safe_name


class ArtifactWriter:
    """Writes deterministic single-image artifacts into output_dir."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def prepare_output_dir(self, output_dir: str | Path) -> Path:
        return ensure_dir(output_dir)

    def save_image(self, image: np.ndarray, path: str | Path) -> str:
        path = Path(path)
        ensure_dir(path.parent)
        try:
            ok = cv2.imwrite(str(path), image)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")
            return str(path)
        except Exception as exc:
            raise PipelineStageError(ARTIFACT_SAVE_ERROR, "Failed to save image artifact", {"path": str(path), "error": str(exc)}) from exc

    def copy_source_ref(self, source_path: str | Path, output_dir: str | Path) -> str:
        output_dir = Path(output_dir)
        dst = output_dir / "source_ref.jpg"
        try:
            ensure_dir(dst.parent)
            shutil.copy2(source_path, dst)
            return str(dst)
        except Exception as exc:
            raise PipelineStageError(ARTIFACT_SAVE_ERROR, "Failed to copy source reference image", {"error": str(exc)}) from exc

    def save_face(self, face_image: np.ndarray, output_dir: str | Path) -> str:
        return self.save_image(face_image, Path(output_dir) / "face.jpg")

    def save_yolo_debug(self, debug_image: np.ndarray, output_dir: str | Path) -> str:
        return self.save_image(debug_image, Path(output_dir) / "yolo_debug.jpg")

    def save_anchor(self, anchor_image: np.ndarray, output_dir: str | Path) -> str:
        return self.save_image(anchor_image, Path(output_dir) / "anchor.jpg")

    def save_recognized_anchor(self, anchor_image: np.ndarray, output_dir: str | Path, student_number: str | None) -> str:
        name = safe_name(student_number, "UNKNOWN")
        return self.save_image(anchor_image, Path(output_dir) / f"recognized_anchor__{name}.jpg")

    def save_result_json(self, result: PipelineResult, output_dir: str | Path) -> str:
        path = Path(output_dir) / "result.json"
        write_json(path, result.to_dict())
        return str(path)

    def save_debug_json(self, debug: dict[str, Any], output_dir: str | Path) -> str:
        path = Path(output_dir) / "debug.json"
        write_json(path, debug)
        return str(path)
