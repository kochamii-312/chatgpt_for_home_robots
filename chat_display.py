import html
from typing import Iterable, Mapping, Any, Optional

import streamlit as st

_ROLE_LABELS = {
    "assistant": "ロボット",
    "user": "ユーザー",
    "tool": "ツール",
}


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return html.escape(content).replace("\n", "<br>")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(html.escape(item.get("text", "")))
                elif item_type == "image_url":
                    url = ""
                    image_payload = item.get("image_url")
                    if isinstance(image_payload, dict):
                        url = image_payload.get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        parts.append("📷 ローカル画像が添付されています。")
                    elif url:
                        safe_url = html.escape(url)
                        parts.append(
                            f"📷 画像: <a href=\"{safe_url}\" target=\"_blank\">リンクを開く</a>"
                        )
                    else:
                        parts.append("📷 画像が添付されています。")
                else:
                    parts.append(html.escape(str(item)))
            else:
                parts.append(html.escape(str(item)))
        return "<br>".join(parts)
    if content is None:
        return ""
    return html.escape(str(content)).replace("\n", "<br>")


def _format_message(role: Optional[str], content: Any) -> str:
    normalized = _normalize_content(content)
    role_key = role or "unknown"
    label = _ROLE_LABELS.get(role_key, role_key)
    classes = ["custom-chat-message"]
    if role_key in ("assistant", "user", "tool"):
        classes.append(f"custom-chat-{role_key}")
    else:
        classes.append("custom-chat-other")
    return (
        f"<div class='{' '.join(classes)}'>"
        f"<div class='custom-chat-role'>{html.escape(str(label))}</div>"
        f"<div class='custom-chat-content'>{normalized}</div>"
        "</div>"
    )


def render_chat_history(
    messages: Iterable[Mapping[str, Any]],
    *,
    greeting: Optional[str] = None,
    height: int = 400,
) -> None:
    """Render chat messages inside a scrollable container."""

    if "__chat_history_style_injected" not in st.session_state:
        st.session_state["__chat_history_style_injected"] = True
        st.markdown(
            """
            <style>
            .custom-chat-history {
                border: 1px solid #dcdcdc;
                border-radius: 0.75rem;
                background: #f9f9f9;
                padding: 1rem;
            }
            .custom-chat-message {
                margin-bottom: 0.75rem;
                padding: 0.5rem 0.75rem;
                border-radius: 0.5rem;
                background: white;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            }
            .custom-chat-message:last-child {
                margin-bottom: 0;
            }
            .custom-chat-role {
                font-weight: 600;
                margin-bottom: 0.25rem;
                color: #444;
            }
            .custom-chat-assistant {
                border-left: 4px solid #4c8bf5;
            }
            .custom-chat-user {
                border-left: 4px solid #34a853;
            }
            .custom-chat-tool {
                border-left: 4px solid #fbbc05;
            }
            .custom-chat-other {
                border-left: 4px solid #999;
            }
            .custom-chat-content {
                color: #1f1f1f;
                line-height: 1.5;
                word-break: break-word;
            }
            .custom-chat-empty {
                color: #777;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    items = []
    if greeting:
        items.append(_format_message("assistant", greeting))

    for msg in messages:
        role = msg.get("role") if isinstance(msg, Mapping) else None
        if role == "system":
            continue
        content = msg.get("content") if isinstance(msg, Mapping) else None
        items.append(_format_message(role, content))

    if not items:
        items.append("<div class='custom-chat-empty'>まだメッセージはありません。</div>")

    safe_height = max(200, height)
    history_html = (
        f"<div class='custom-chat-history' style='height: {safe_height}px; overflow: auto;'>"
        + "".join(items)
        + "</div>"
    )
    st.markdown(history_html, unsafe_allow_html=True)
