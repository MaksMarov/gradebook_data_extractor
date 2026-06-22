from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "on", "y", "gpu", "cuda"}
FALSE_VALUES = {"0", "false", "no", "off", "n", "cpu", "none"}
AUTO_VALUES = {"", "auto", "default"}


@dataclass(slots=True, frozen=True)
class TorchCudaInfo:
    status: str
    torch_available: bool
    cuda_available: bool
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    device_count: int = 0
    device_names: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "device_count": self.device_count,
            "device_names": list(self.device_names),
            "error": self.error,
        }


def torch_cuda_info() -> TorchCudaInfo:
    """Return CUDA availability without making the rest of the package depend on torch at import time."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        return TorchCudaInfo(
            status="error",
            torch_available=False,
            cuda_available=False,
            error=f"torch import failed: {type(exc).__name__}: {exc}",
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        names = tuple(str(torch.cuda.get_device_name(i)) for i in range(device_count))
        return TorchCudaInfo(
            status="ok" if cuda_available else "cpu",
            torch_available=True,
            cuda_available=cuda_available,
            torch_version=str(getattr(torch, "__version__", "")) or None,
            torch_cuda_version=str(getattr(torch.version, "cuda", "")) or None,
            device_count=device_count,
            device_names=names,
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        return TorchCudaInfo(
            status="error",
            torch_available=True,
            cuda_available=False,
            torch_version=str(getattr(torch, "__version__", "")) or None,
            error=f"cuda check failed: {type(exc).__name__}: {exc}",
        )


def parse_boolish(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES or normalized in AUTO_VALUES:
        return False
    return default


def is_auto(value: object) -> bool:
    return value is None or str(value).strip().lower() in AUTO_VALUES


def normalize_device(value: object) -> str:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    if normalized in AUTO_VALUES:
        return "auto"
    if normalized in {"gpu", "cuda", "cuda:0", "0"}:
        return "cuda"
    if normalized.startswith("cuda"):
        return normalized
    if normalized in {"cpu", "none"}:
        return "cpu"
    return normalized


def resolve_compute_device(requested: object = "auto") -> str:
    requested_device = normalize_device(requested)
    if requested_device != "auto":
        return requested_device

    info = torch_cuda_info()
    return "cuda" if info.cuda_available else "cpu"


def resolve_yolo_device(compute_device: object = "auto", yolo_device: object = None) -> str | None:
    """Resolve Ultralytics device argument.

    Ultralytics accepts values like `cpu`, `0`, `0,1`, or `cuda:0` through
    the predict `device` argument. Returning None means library default.
    """
    if not is_auto(yolo_device):
        value = str(yolo_device).strip().lower()
        if value in {"gpu", "cuda", "cuda:0"}:
            return "0"
        return value

    device = resolve_compute_device(compute_device)
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1] or "0"
    return None


def resolve_easyocr_gpu(compute_device: object = "auto", easyocr_gpu: object = "auto") -> bool | str:
    """Resolve EasyOCR Reader gpu argument.

    EasyOCR accepts a boolean and also supports a device string. We keep string
    values when the operator explicitly provides them, otherwise auto maps to a
    boolean based on CUDA availability.
    """
    if isinstance(easyocr_gpu, bool):
        return easyocr_gpu

    value = "auto" if easyocr_gpu is None else str(easyocr_gpu).strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    if value not in AUTO_VALUES:
        return value

    device = resolve_compute_device(compute_device)
    return device != "cpu"
