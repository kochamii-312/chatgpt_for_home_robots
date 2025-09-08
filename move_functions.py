import os
import streamlit as st

DEFAULT_IMAGE_DIR = "images"

# 実際のロボットの代わりに動作をプリント
def move_to(room_name):
    show_room_image(room_name)
    return f"Moved to {room_name}"

def pick_object(obj):
    return f"Picked up {obj}"

def place_object_next_to(obj, target):
    return f"Placed {obj} next to {target}"

def place_object_on(obj, target):
    return f"Placed {obj} on {target}"

def place_object_in(obj, target):
    return f"Placed {obj} in {target}"

def detect_object(obj):
    return f"Detect {obj}"

def search_about(obj):
    return f"Searched about {obj}"

def push(obj):
    return f"Pushed {obj}"

def say(text):
    return f"Said {text}"

def _room_to_path(room_name: str) -> str:
    fname = f"{room_name.lower()}.png"  # "KITCHEN" -> "kitchen.png"
    image_dir = DEFAULT_IMAGE_DIR
    house = st.session_state.get("selected_house")
    if house:
        image_dir = os.path.join(image_dir, house)
    return os.path.join(image_dir, fname)

def show_room_image(room_name: str) -> str:
    """
    Plan 実行時に呼ばれる表示関数。
    ローカルにある部屋画像（例: images/house1/kitchen.png）を表示する。
    """
    path = get_room_image_path(room_name)
    if os.path.exists(path):
        st.image(path, caption=f"{room_name} (local)")
        return f"Displayed local image for {room_name}: {path}"
    else:
        # ローカルに無い場合は控えめにメッセージ（必要なら公開URLのフォールバック実装も可能）
        st.warning(f"画像が見つかりません: {path}")
        return f"No local image found for {room_name}"

def get_room_image_path(room_name: str) -> str:
    house = st.session_state.get("selected_house")
    if house:
        candidate = os.path.join(DEFAULT_IMAGE_DIR, house, f"{room_name.lower()}.png")
        if os.path.exists(candidate):
            return candidate
    return _room_to_path(room_name)
