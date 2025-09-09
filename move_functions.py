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
    subfolder = st.session_state.get("selected_subfolder")
    if house:
        image_dir = os.path.join(image_dir, house)
        if subfolder:
            image_dir = os.path.join(image_dir, subfolder)
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
    """Return path to a room image, searching available image folders.

    The user may choose a specific house via ``selected_house`` in the
    session state.  When not specified (or when the requested image does
    not exist for the selected house), the function now searches all
    subdirectories under ``images`` and falls back to the first match.
    This allows images to be displayed even when no house is selected in
    the UI.
    """

    file_name = f"{room_name.lower()}.png"
    house = st.session_state.get("selected_house")
    subfolder = st.session_state.get("selected_subfolder")

    # Directories to search, in order of priority
    search_dirs = []
    if house:
        base_dir = os.path.join(DEFAULT_IMAGE_DIR, house)
        if subfolder:
            search_dirs.append(os.path.join(base_dir, subfolder))
        search_dirs.append(base_dir)

    # If no house is selected, search all house directories first
    if not house:
        for d in os.listdir(DEFAULT_IMAGE_DIR):
            subdir = os.path.join(DEFAULT_IMAGE_DIR, d)
            if os.path.isdir(subdir):
                search_dirs.append(subdir)

    # Finally, look in the top-level images directory
    search_dirs.append(DEFAULT_IMAGE_DIR)

    for directory in search_dirs:
        candidate = os.path.join(directory, file_name)
        if os.path.exists(candidate):
            return candidate

    # Return the default path even if it doesn't exist; caller will warn
    return os.path.join(search_dirs[0] if search_dirs else DEFAULT_IMAGE_DIR, file_name)
