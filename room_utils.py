import os
from typing import Set
import streamlit as st
from api import build_bootstrap_user_message
from move_functions import show_room_image, get_room_image_path

ROOM_TOKENS = ["BEDROOM", "KITCHEN", "DINING", "LIVING", "BATHROOM", "和室", "HALL", "LDK"]

def detect_rooms_in_text(text: str) -> Set[str]:
    """Return a set of room tokens found in text."""
    found: Set[str] = set()
    up = (text or "").upper()
    for r in ROOM_TOKENS:
        if r == "和室":
            if "和室" in (text or ""):
                found.add(r)
        else:
            if r in up:
                found.add(r)
    return found

def attach_images_for_rooms(rooms: Set[str], show_in_ui: bool = True) -> None:
    """Attach room images for new rooms to the conversation context and optionally display them."""
    if "sent_room_images" not in st.session_state:
        st.session_state.sent_room_images = set()
    new_rooms = [r for r in rooms if r not in st.session_state.sent_room_images]
    if not new_rooms:
        return
    local_paths = []
    for room in new_rooms:
        img_path = get_room_image_path(room)
        if os.path.exists(img_path):
            if show_in_ui:
                show_room_image(room)
            local_paths.append(img_path)
            st.session_state.sent_room_images.add(room)
        else:
            st.warning(f"{room} の画像が見つかりません: {img_path}")
    if local_paths:
        st.session_state["context"].append(
            build_bootstrap_user_message(
                text=f"Here are room images for: {', '.join(new_rooms)}. Use them for scene understanding and disambiguation.",
                local_image_paths=local_paths,
            )
        )
