
# habitat_bridge.py
# A drop-in backend for the 2D stubs, using Meta's Habitat-Sim.
# It mirrors the methods used by move_functions.py:
#   - move(direction, distance_m)
#   - rotate(direction, angle_deg)
#   - go_to_location(location_name)
#   - stop()
#   - get_pose(), reset_log(), action_log
#
# Usage:
#   Set environment variable USE_HABITAT=1 and HABITAT_SCENE=/path/to/scene.glb
#   Optionally HABITAT_LOCATIONS=/path/to/locations.json ({"kitchen":[x,y,yaw_deg], ...})
#
# Notes:
#   - This adapter directly updates the agent state (continuous move/rotate).
#   - Replace with discrete actions if you prefer (sim.step("move_forward")).
#   - Coordinate units are meters; yaw_deg is CCW around +Z, 0° = +X (Habitat default).

from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

action_log: List[str] = []

def _log(msg: str):
    action_log.append(msg)

def reset_log():
    action_log.clear()

try:
    import habitat_sim
    from habitat_sim.utils import common as hutils
    import numpy as np
    _HABITAT_OK = True
except Exception as e:
    _HABITAT_OK = False
    _IMPORT_ERR = e

@dataclass
class Pose2D:
    x: float
    y: float
    yaw_deg: float

class HabitatSimAdapter:
    def __init__(self, scene_path: str, locations_json: Optional[str] = None):
        if not _HABITAT_OK:
            raise RuntimeError(f"Habitat-Sim import failed: {_IMPORT_ERR}")
        if not scene_path or not os.path.exists(scene_path):
            raise FileNotFoundError(f"HABITAT_SCENE not found: {scene_path}")

        # --- Simulator config ---
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path

        # Sensor specs: RGB only (you can add depth/semantic if needed)
        sensor_specs = []
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [480, 640]
        rgb_sensor.position = [0.0, 1.0, 0.0]
        rgb_sensor.orientation = [0.0, 0.0, 0.0]
        sensor_specs.append(rgb_sensor)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        # Default step parameters (used only if you switch to discrete actions)
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self.agent = self.sim.initialize_agent(0)
        self.locations: Dict[str, Tuple[float, float, float]] = {}

        if locations_json and os.path.exists(locations_json):
            try:
                with open(locations_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Expect {"kitchen":[x,y,yaw_deg], ...}
                for k, v in data.items():
                    if isinstance(v, list) and len(v) >= 2:
                        yaw = float(v[2]) if len(v) >= 3 else 0.0
                        self.locations[k] = (float(v[0]), float(v[1]), yaw)
            except Exception as e:
                _log(f"locations.json 読み込み失敗: {e}")

        # Default locations (meters)
        self.locations.update({
            "玄関": (0.0, 0.0, 0.0),
            "リビング": (1.5, 0.5, 0.0),
            "キッチン": (3.0, 0.5, 0.0),
            "ダイニング": (2.5, -0.8, 0.0),
            "机": (1.2, 1.2, 0.0),
            "テーブル": (2.4, -0.6, 0.0),
            "bedroom": (-1.0, 0.0, 0.0),
            "kitchen": (3.0, 0.5, 0.0),
            "dining": (2.5, -0.8, 0.0),
            "living": (1.5, 0.5, 0.0),
        })

        _log(f"Habitat initialized: {scene_path}")

    # --- Utilities ---
    def _get_state(self):
        return self.agent.get_state()

    def _set_state(self, position_xyz, rotation_quat):
        st = self._get_state()
        st.position = position_xyz
        st.rotation = rotation_quat
        self.agent.set_state(st)

    # --- Public API ---
    def get_pose(self) -> Pose2D:
        st = self._get_state()
        x, y, z = st.position  # Habitat uses [x, y, z]
        # Derive yaw from quaternion
        q = st.rotation
        # Convert quaternion to yaw (rotation around +Y is "up" in Habitat)
        # Habitat's up axis is Y; we map yaw about Y to our yaw_deg.
        # Extract yaw from quaternion
        # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        ysqr = q.y * q.y
        t3 = +2.0 * (q.w * q.y + q.x * q.z)
        t4 = +1.0 - 2.0 * (ysqr + q.z * q.z)
        yaw_rad = math.atan2(t3, t4)
        yaw_deg = math.degrees(yaw_rad)
        return Pose2D(x=float(x), y=float(z), yaw_deg=float(yaw_deg))

    def set_pose(self, x: float, y: float, yaw_deg: float):
        st = self._get_state()
        # Map our (x, y) to Habitat (x, z), keep current height
        pos = st.position
        new_pos = np.array([float(x), pos[1], float(y)], dtype=np.float32)
        # Yaw about Y axis
        quat = hutils.quat_from_angle_axis(math.radians(yaw_deg), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self._set_state(new_pos, quat)
        _log(f"set_pose(x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f}°)")

    def rotate(self, direction: str, angle_deg: float):
        pose = self.get_pose()
        sign = +1 if direction.lower() in ["left", "左"] else -1
        new_yaw = (pose.yaw_deg + sign * float(angle_deg)) % 360.0
        self.set_pose(pose.x, pose.y, new_yaw)
        _log(f"rotate({direction}, {angle_deg}deg) → yaw={new_yaw:.1f}°")

    def move(self, direction: str, distance_m: float):
        pose = self.get_pose()
        heading = math.radians(pose.yaw_deg)
        dx_fwd = math.cos(heading) * distance_m
        dy_fwd = math.sin(heading) * distance_m
        dx_lat = -math.sin(heading) * distance_m
        dy_lat =  math.cos(heading) * distance_m

        d = direction.lower()
        if d in ["forward", "前", "前進"]:
            dx, dy = dx_fwd, dy_fwd
        elif d in ["backward", "後ろ", "後退", "back"]:
            dx, dy = -dx_fwd, -dy_fwd
        elif d in ["left", "左"]:
            dx, dy = dx_lat, dy_lat
        elif d in ["right", "右"]:
            dx, dy = -dx_lat, -dy_lat
        else:
            _log(f"move({direction}, {distance_m}m) ✖ 未知のdirection")
            return "NG: unknown direction"

        self.set_pose(pose.x + dx, pose.y + dy, pose.yaw_deg)
        _log(f"move({direction}, {distance_m}m) → ({pose.x+dx:.2f}, {pose.y+dy:.2f})")
        return (pose.x + dx, pose.y + dy)

    def go_to_location(self, name: str):
        key = name if name in self.locations else name.lower()
        if key not in self.locations:
            _log(f"go_to_location('{name}') ✖ 未登録の場所")
            return f"NG: location '{name}' not registered"
        x, y, yaw = self.locations[key]
        self.set_pose(x, y, yaw)
        _log(f"go_to_location('{name}') → ({x:.2f}, {y:.2f}) / yaw={yaw:.1f}°")
        return (x, y, yaw)

    def stop(self):
        _log("stop()")
        return "stopped"


# Singleton factory
_ADAPTER: Optional[HabitatSimAdapter] = None

def get_adapter() -> HabitatSimAdapter:
    global _ADAPTER
    if _ADAPTER is not None:
        return _ADAPTER

    scene = os.getenv("HABITAT_SCENE", "")
    locs = os.getenv("HABITAT_LOCATIONS", "")
    _ADAPTER = HabitatSimAdapter(scene, locs if locs else None)
    return _ADAPTER
