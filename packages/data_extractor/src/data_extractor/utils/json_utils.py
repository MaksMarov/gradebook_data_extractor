from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """
    Приводит объект к виду, который можно безопасно сериализовать через json.

    Обрабатывает:
    - numpy int/float/bool
    - numpy arrays
    - pathlib.Path
    - dataclass
    - dict/list/tuple/set
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return to_jsonable(asdict(value))

    # numpy scalar: np.int32, np.float32, np.bool_
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass

    # numpy array
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    return str(value)


def dumps_json(value: Any, **kwargs: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, **kwargs)


def write_json(path: str | Path, value: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dumps_json(value, indent=indent),
        encoding="utf-8",
    )
