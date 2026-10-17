"""Small file and directory helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory, and its parents, if it does not exist.

    Args:
        path: Directory to create.

    Returns:
        The directory as a ``Path``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Write a dictionary to a JSON file with sorted keys.

    Args:
        path: Output file. Missing parent directories are created.
        obj: A JSON-serializable dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
