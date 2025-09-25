"""Utilities for persisting image and task set configurations.

This module centralises the read/write logic so that multiple Streamlit pages
can share the same storage format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_DATA_PATH = Path("json/image_task_sets.json")


def load_image_task_sets() -> Dict[str, Dict[str, Any]]:
    """Load all stored image/task sets.

    Returns an empty dictionary when the storage file does not exist or is
    malformed. The function is resilient to partial corruption and will ignore
    entries that are not dictionaries.
    """

    if not _DATA_PATH.exists():
        return {}

    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(raw, dict):
        return {}

    cleaned: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            cleaned[str(key)] = value
    return cleaned


def save_image_task_sets(task_sets: Dict[str, Dict[str, Any]]) -> None:
    """Persist the provided task sets to disk."""

    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATA_PATH.write_text(
        json.dumps(task_sets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_image_task_set(name: str, payload: Dict[str, Any]) -> None:
    """Create or update a single task set entry."""

    task_sets = load_image_task_sets()
    task_sets[name] = payload
    save_image_task_sets(task_sets)


def delete_image_task_set(name: str) -> None:
    """Remove a task set from storage if it exists."""

    task_sets = load_image_task_sets()
    if name in task_sets:
        task_sets.pop(name)
        save_image_task_sets(task_sets)
