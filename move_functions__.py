
# move_functions.py (Habitat-aware)
# Unified API for robot actions. Chooses backend:
#   - If USE_HABITAT=1 and habitat_sim is importable → use HabitatSimAdapter
#   - Else fall back to our SimpleSim stub (sim_adapter.SIM)

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

def move(direction: str, distance_m: float):
    return _adapter.move(direction, float(distance_m))

def rotate(direction: str, angle_deg: float):
    _adapter.rotate(direction, float(angle_deg))
    p = get_pose()
    return f"yaw={p[2]:.1f}deg"

def go_to_location(location_name: str):
    return _adapter.go_to_location(location_name)

def stop():
    return _adapter.stop()

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
