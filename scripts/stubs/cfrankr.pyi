from typing import Any, List


class Affine:
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, a: float = 0.0, b: float = 0.0, c: float = 0.0):
        ...

    def inverse(self) -> Affine:
        ...

    def get_inner_random(self) -> Affine:
        ...

    def translation(self) -> List[float]:
        ...

    def __mul__(self, a: Affine) -> Affine:
        ...


class MotionData:
    did_break: bool

    @classmethod
    def with_dynamics(cls, dynamics_rel: float) -> MotionData:
        ...

    @classmethod
    def with_z_force_condition(cls, force: float) -> MotionData:
        ...

    @classmethod
    def with_xy_force_condition(cls, force: float) -> MotionData:
        ...


class Waypoint:
    target_affine: Affine
    reference_type: Any

    ReferenceType: Any

    def __init__(self, target_affine: Affine, reference_type: Any):
        ...


class Robot:
    is_moving: bool
    has_reflex_error: bool

    def __init__(self, address: str, value=1.0):
        ...

    def current_pose(self, frame: Affine) -> Affine:
        ...

    def recover_from_errors(self) -> bool:
        ...

    def move_cartesian(self, frame: Affine, affine: Affine, md: MotionData) -> bool:
        ...

    def move_relative_cartesian(self, frame: Affine, affine: Affine, md: MotionData) -> bool:
        ...

    def move_joints(self, joint_values: List[float], md: MotionData) -> bool:
        ...

    def move_waypoints_cartesian(self, frame: Affine, waypoints: List[Waypoint], md: MotionData) -> bool:
        ...


class Gripper:
    max_width: float

    def __init__(self, address: str, value: float):
        ...

    def width(self) -> float:
        ...

    def homing(self) -> bool:
        ...

    def stop(self) -> bool:
        ...

    def is_grasping(self) -> bool:
        ...

    def move(self, width: float) -> None:
        ...

    def clamp(self) -> bool:
        ...

    def release(self, width: float) -> None:
        ...


