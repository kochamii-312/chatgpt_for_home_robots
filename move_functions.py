# move_functions.py
# Thin wrappers that the Streamlit pages import.
# They call into sim_adapter.SIM so you can later swap the backend.

import os
from typing import Tuple, Union

_BACKEND = "simple"

try:
    if os.getenv("USE_HABITAT", "0") == "1":
        from habitat_bridge import get_adapter, reset_log, action_log
        _adapter = get_adapter()
        _BACKEND = "habitat"
    else:
        raise ImportError("USE_HABITAT != 1")
except Exception:
    from sim_adapter import SIM as _adapter
    from sim_adapter import SimpleSim as _SimpleSim
    from sim_adapter import Pose2D as _Pose2D
    def reset_log(): _adapter.reset_log()
    def action_log(): return _adapter.action_log

def move(direction: str, distance_m: float) -> Union[Tuple[float, float], str]:
    """Move in a relative direction by distance (m)."""
    return _adapter.move(direction, float(distance_m))

def rotate(direction: str, angle_deg: float):
    """Rotate left/right by angle (deg)."""
    _adapter.rotate(direction, float(angle_deg))
    p = get_pose()
    return f"yaw={p[2]:.1f}deg"

def go_to_location(location_name: str):
    """Teleport to a named semantic location (stub for path planning)."""
    return _adapter.go_to_location(location_name)

def stop() -> str:
    """Stop current motion (no-op in this stub)."""
    return _adapter.stop()


def move_to(room_name: str) -> str:
    """Move robot to the specified room."""
    return _adapter.move_to(room_name)

def pick_object(object: str) -> str:
    """Pick up the specified object."""
    return _adapter.pick_object(object)

def place_object_next_to(object: str, target: str) -> str:
    """Place the object next to the target."""
    return _adapter.place_object_next_to(object, target)

def place_object_on(object: str, target: str) -> str:
    """Place the object on the target."""
    return _adapter.place_object_on(object, target)

def place_object_in(object: str, target: str) -> str:
    """Place the object in the target."""
    return _adapter.place_object_in(object, target)

def detect_object(object: str) -> str:
    """Detect the specified object using YOLO."""
    return _adapter.detect_object(object)

def search_about(object: str) -> str:
    """Search information about the specified object."""
    return _adapter.search_about(object)

def push(object: str) -> str:
    """Push the specified object."""
    return _adapter.push(object)

def say(text: str) -> str:
    """Speak the specified text."""
    return _adapter.say(text)


def get_pose() -> tuple[float, float, float]:
    p = _adapter.get_pose()
    # normalize for both backends
    if isinstance(p, tuple) and len(p) == 3:
        return p  # already tuple
    # object with x,y,yaw_deg
    return (p.x, p.y, p.yaw_deg)

def get_log():
    try:
        return list(action_log())
    except TypeError:
        return list(action_log)

def reset_log_wrapper():
    reset_log()
