from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import streamlit as st

from image_task_sets import resolve_image_paths
from utils.firebase_utils import save_document

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IMAGE_DIRECTORIES = {
    "DINING": _PROJECT_ROOT / "images" / "dining",
    "FLOWER": _PROJECT_ROOT / "images" / "flower",
}
_IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif")


def _normalise_candidate_paths(candidates: Any, limit: int) -> list[str]:
    """Return a list of path strings derived from ``candidates``."""

    if not isinstance(candidates, Iterable) or isinstance(candidates, (str, bytes)):
        return []

    paths: list[str] = []
    for item in candidates:
        if item:
            paths.append(str(item))
        if len(paths) >= limit:
            break
    return paths


def _determine_image_directory(
    selected_prompt: dict[str, Any] | None, prompt_label: str | None
) -> Path | None:
    """Return the image directory that best matches the current task."""

    if isinstance(selected_prompt, dict):
        explicit = selected_prompt.get("image_directory")
        if explicit:
            candidate = Path(explicit)
            if not candidate.is_absolute():
                candidate = (_PROJECT_ROOT / candidate).resolve()
            if candidate.exists():
                return candidate

    label_candidates: list[str] = []

    if prompt_label:
        label_candidates.append(str(prompt_label))

    if isinstance(selected_prompt, dict):
        for key in ("task", "taskinfo"):
            value = selected_prompt.get(key)
            if isinstance(value, str) and value.strip():
                label_candidates.append(value)

    for label in label_candidates:
        upper_label = label.upper()
        if "DINING" in upper_label or any(
            keyword in label for keyword in ("夕食", "テーブル", "食事", "ディナー")
        ):
            return _IMAGE_DIRECTORIES.get("DINING")
        if "FLOWER" in upper_label or any(
            keyword in label for keyword in ("花束", "花", "フラワー")
        ):
            return _IMAGE_DIRECTORIES.get("FLOWER")

    return None


def _collect_directory_images(directory: Path | None, limit: int) -> list[str]:
    """Return image paths from ``directory`` up to ``limit`` entries."""

    if directory is None:
        return []

    images: list[str] = []
    if directory.exists():
        seen: set[str] = set()
        for pattern in _IMAGE_PATTERNS:
            for path in sorted(directory.glob(pattern)):
                path_str = str(path)
                if path_str not in seen:
                    images.append(path_str)
                    seen.add(path_str)
                if len(images) >= limit:
                    break
            if len(images) >= limit:
                break
    return images


def _to_storage_path(path: str) -> str:
    """Convert an absolute path to a project-relative string when possible."""

    try:
        resolved = Path(path).resolve()
    except OSError:
        return path

    try:
        return str(resolved.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def render_task_completion_image_choice(
    *,
    selected_prompt: dict[str, Any] | None,
    prompt_label: str | None,
    memo_state_key: str,
    memo_input_key: str,
    memo_save_key: str,
    instruction_text: str | None = None,
    max_images: int = 5,
) -> None:
    """Render an image selection UI for the task completion memo."""

    st.session_state.setdefault(memo_state_key, "")

    instruction = instruction_text or (
        "上記のタスクが完了した状態を想像し、写真からイメージに近いものを選んでください。"
    )
    st.markdown(f"**{instruction}**")

    candidates: Sequence[str] = []
    if isinstance(selected_prompt, dict):
        candidates = selected_prompt.get("image_candidates") or []

    image_directory = _determine_image_directory(selected_prompt, prompt_label)
    if image_directory is not None:
        candidate_paths = _collect_directory_images(image_directory, max_images)
    else:
        candidate_paths = _normalise_candidate_paths(candidates, max_images)

    existing, missing = resolve_image_paths(candidate_paths)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        st.warning(
            "以下の画像を読み込めませんでした。"
            "設定を確認してください:\n" + missing_list
        )

    if not existing:
        st.info("表示できる画像がありません。管理画面で画像候補を設定してください。")
        st.session_state.setdefault(memo_input_key, "")
        if st.button("保存", key=memo_save_key, type="primary", disabled=True):
            pass
        return

    current_saved = st.session_state.get(memo_state_key, "")
    if memo_input_key not in st.session_state:
        default_value = current_saved if current_saved in existing else existing[0]
        st.session_state[memo_input_key] = default_value
    elif st.session_state[memo_input_key] not in existing:
        st.session_state[memo_input_key] = existing[0]

    option_labels = {
        path: f"画像{index}"
        for index, path in enumerate(existing, start=1)
    }

    columns = st.columns(len(existing))
    for path, column in zip(existing, columns, strict=False):
        label = option_labels[path]
        with column:
            column.markdown(f"**{label.split('（')[0]}**")
            column.image(path, use_container_width=True)
            column.caption(Path(path).name)

    st.radio(
        "どれが一番イメージに合っていますか？",
        existing,
        key=memo_input_key,
        format_func=lambda path: option_labels.get(path, path),
    )

    status_placeholder = st.empty()

    if st.button("保存", key=memo_save_key, type="primary"):
        st.session_state[memo_state_key] = st.session_state.get(memo_input_key, "")
        selected_path = st.session_state.get(memo_input_key, "")
        storage_path = _to_storage_path(selected_path)
        available_images = [_to_storage_path(path) for path in existing]
        payload = {
            "selected_image": storage_path,
            "available_images": available_images,
            "prompt_label": prompt_label or "",
            "prompt_group": st.session_state.get("prompt_group", ""),
            "task": selected_prompt.get("task", "") if isinstance(selected_prompt, dict) else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if image_directory is not None:
            payload["image_directory"] = _to_storage_path(str(image_directory))
        payload["instruction_text"] = instruction

        try:
            save_document("task_completion_image_choices", payload)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            status_placeholder.error(f"Firestoreへの保存に失敗しました: {exc}")
        else:
            status_placeholder.success("選択を保存しました。")
