from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import streamlit as st

from image_task_sets import resolve_image_paths


def _normalise_candidate_paths(candidates: Any, limit: int) -> list[str]:
    """Return a list of path strings derived from ``candidates``.

    ``candidates`` may be any iterable containing objects that can be
    converted to strings.  Only the first ``limit`` entries are returned,
    preserving order and skipping falsy values.
    """

    if not isinstance(candidates, Iterable) or isinstance(candidates, (str, bytes)):
        return []

    paths: list[str] = []
    for item in candidates:
        if item:
            paths.append(str(item))
        if len(paths) >= limit:
            break
    return paths


def render_task_completion_image_choice(
    *,
    selected_prompt: dict[str, Any] | None,
    memo_state_key: str,
    memo_input_key: str,
    memo_save_key: str,
    instruction_text: str | None = None,
    max_images: int = 5,
) -> None:
    """Render an image selection UI for the task completion memo.

    Args:
        selected_prompt: Prompt metadata that may contain ``image_candidates``.
        memo_state_key: Session state key used to persist the saved choice.
        memo_input_key: Session state key used by the radio widget.
        memo_save_key: Session state key used by the save button.
        instruction_text: Optional text shown above the image grid.
        max_images: Maximum number of images to display.
    """

    st.session_state.setdefault(memo_state_key, "")

    instruction = instruction_text or (
        "上記のタスクが完了した状態を想像し、写真からイメージに近いものを選んでください。"
    )
    st.markdown(f"**{instruction}**")

    candidates: Sequence[str] = []
    if isinstance(selected_prompt, dict):
        candidates = selected_prompt.get("image_candidates") or []

    candidate_paths = _normalise_candidate_paths(candidates, max_images)
    existing, missing = resolve_image_paths(candidate_paths)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        st.warning(
            "以下の画像を読み込めませんでした。" "設定を確認してください:\n" + missing_list
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

    if st.button("保存", key=memo_save_key, type="primary"):
        st.session_state[memo_state_key] = st.session_state.get(memo_input_key, "")
        st.success("選択を保存しました。")
