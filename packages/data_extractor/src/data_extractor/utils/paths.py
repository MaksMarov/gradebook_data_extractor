from __future__ import annotations

import re
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(value: str | None, fallback: str = "UNKNOWN") -> str:
    value = str(value or fallback).strip()
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_\-.]+", "_", value)
    return value or fallback


def unique_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    root = path.with_suffix("")
    suffix = path.suffix
    i = 2
    while True:
        candidate = Path(f"{root}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1
