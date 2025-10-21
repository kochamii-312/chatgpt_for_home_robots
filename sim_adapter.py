
# sim_adapter.py
# A minimal, self-contained "simulator" to let you run and visualize robot motions
# from LLM plans inside Streamlit without external engines. It tracks pose (x,y,yaw)
# and named locations. Replace with Habitat/Isaac/Gazebo bindings later if needed.

from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple, Optional

@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0  # 0 deg = +X axis, CCW positive

@dataclass
class SimpleSim:
    # World scale: meters
    pose: Pose2D = field(default_factory=Pose2D)
    # Pre-registered semantic locations (replace with map-based nav later)
    locations: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "玄関": (0.0, 0.0),
        "リビング": (1.5, 0.5),
        "キッチン": (3.0, 0.5),
        "ダイニング": (2.5, -0.8),
        "机": (1.2, 1.2),
        "テーブル": (2.4, -0.6),
        "bedroom": (-1.0, 0.0),
        "kitchen": (3.0, 0.5),
        "dining": (2.5, -0.8),
        "living": (1.5, 0.5),
    })
    action_log: List[str] = field(default_factory=list)
    stopped: bool = False

    def _log(self, msg: str):
        self.action_log.append(msg)

    def reset_log(self):
        self.action_log.clear()

    def get_pose(self) -> Pose2D:
        return self.pose

    def set_pose(self, x: float, y: float, yaw_deg: float):
        self.pose = Pose2D(x, y, yaw_deg)
        self._log(f"set_pose(x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f}°)")

    def rotate(self, direction: str, angle_deg: float):
        sign = +1 if direction.lower() in ["left", "左"] else -1
        self.pose.yaw_deg = (self.pose.yaw_deg + sign * angle_deg) % 360.0
        self._log(f"rotate({direction}, {angle_deg}deg) → yaw={self.pose.yaw_deg:.1f}°")

    def move(self, direction: str, distance_m: float):
        # forward/back move along heading; left/right = strafe
        heading = math.radians(self.pose.yaw_deg)
        dx_fwd = math.cos(heading) * distance_m
        dy_fwd = math.sin(heading) * distance_m
        dx_lat = -math.sin(heading) * distance_m
        dy_lat =  math.cos(heading) * distance_m

        d = direction.lower()
        if d in ["forward", "前", "前進"]:
            dx, dy = dx_fwd, dy_fwd
        elif d in ["backward", "後ろ", "後退"]:
            dx, dy = -dx_fwd, -dy_fwd
        elif d in ["left", "左"]:
            dx, dy = dx_lat, dy_lat
        elif d in ["right", "右"]:
            dx, dy = -dx_lat, -dy_lat
        else:
            self._log(f"move({direction}, {distance_m}m) ✖ 未知のdirection")
            return "NG: unknown direction"

        self.pose.x += dx
        self.pose.y += dy
        self._log(f"move({direction}, {distance_m}m) → ({self.pose.x:.2f}, {self.pose.y:.2f})")
        return (self.pose.x, self.pose.y)

    def go_to_location(self, name: str):
        key = name
        # allow both Japanese and lowercase english keys by normalizing
        if key not in self.locations and key.lower() in self.locations:
            key = key.lower()
        if key not in self.locations:
            self._log(f"go_to_location('{name}') ✖ 未登録の場所")
            return f"NG: location '{name}' not registered"

        x, y = self.locations[key]
        self.pose.x, self.pose.y = x, y
        self._log(f"go_to_location('{name}') → ({x:.2f}, {y:.2f})")
        return (x, y)

    def stop(self):
        self.stopped = True
        self._log("stop()")
        return "stopped"

# A module-level singleton for simplicity
SIM = SimpleSim()
