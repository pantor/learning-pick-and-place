import numpy as np

from actions.action import RobotPose
from cfrankr import Affine
from config import Config
from utils.param import Bin


class Frames:
    camera = Affine(-0.079, -0.0005, 0.011, -np.pi / 4, 0.0, -np.pi)
    gripper = Affine(0.0, 0.0, 0.18, -np.pi / 4, 0.0, -np.pi)

    bin_frames = {
        Bin.Left: Affine(0.480, -0.125, 0.011, np.pi / 2),
        Bin.Right: Affine(0.480, 0.125, 0.011, np.pi / 2),
    }

    bin_joint_values = {
        Bin.Left: [
            -1.8119446041276,
            1.1791089121678,
            1.7571002245448,
            -2.141621800118,
            -1.143369391372,
            1.633046061666,
            -0.432171664388
        ],
        Bin.Right: [
            -1.4637412426804,
            1.0494154046592,
            1.7926908288289,
            -2.283032105735,
            -1.035444400130,
            1.752863485400,
            0.04325164650034
        ],
    }

    release_fastest = Affine(0.480, -0.06, 0.180, np.pi / 2)
    release_midway = Affine(0.480, 0.0, 0.240, np.pi / 2)

    @classmethod
    def get_frame(cls, current_bin: Bin) -> Affine:
        return cls.bin_frames[current_bin]

    @classmethod
    def get_camera_frame(cls, current_bin: Bin, a=0.0, b=0.0, c=0.0, reference_pose=None) -> Affine:
        reference_pose = reference_pose or Affine()
        lateral = Affine(b=b, c=c) * Affine(a=a).inverse()
        return cls.bin_frames[current_bin] * Affine(b=lateral.b, c=lateral.c) * Affine(b=reference_pose.b, c=reference_pose.c) * Config.default_image_pose.inverse()

    @classmethod
    def get_release_frame(cls, current_bin: Bin) -> Affine:
        return Affine(z=Config.move_down_distance_for_release) * cls.bin_frames[current_bin]

    @classmethod
    def get_next_bin(cls, current_bin: Bin) -> Bin:
        return {Bin.Left: Bin.Right, Bin.Right: Bin.Left}[current_bin]

    @classmethod
    def get_pose_in_image(cls, action_pose: RobotPose, image_pose: Affine, reference_pose: Affine = None) -> RobotPose:
        if np.isnan(action_pose.z):
            action_pose.z = -0.35  # [m]

        z_offset = -0.022  # [m]

        image_translation = Affine(image_pose.x, image_pose.y, image_pose.z + z_offset)
        image_rotation = Affine(a=image_pose.a, b=image_pose.b, c=image_pose.c)
        action_translation = Affine(action_pose.x, action_pose.y, action_pose.z)
        action_rotation = Affine(b=action_pose.b, c=action_pose.c) * Affine(a=action_pose.a)
        affine = image_rotation * image_translation.inverse() * action_translation * action_rotation

        if reference_pose:
            reference_rotation = Affine(b=reference_pose.b, c=reference_pose.c).inverse()
            affine = reference_rotation * affine

        return RobotPose(affine=affine, d=action_pose.d)

    @classmethod
    def get_action_pose(cls, action_pose: RobotPose, image_pose: Affine) -> RobotPose:
        action_translation = Affine(action_pose.x, action_pose.y, action_pose.z)
        action_rotation = Affine(a=action_pose.a, b=action_pose.b, c=action_pose.c)
        affine = image_pose * action_translation * action_rotation
        return RobotPose(affine=affine, d=action_pose.d)

    @classmethod
    def is_gripper_frame_safe(cls, current_pose: Affine) -> bool:
        return current_pose.z > 0.16  # [m]

    @classmethod
    def is_camera_frame_safe(cls, camera_frame) -> bool:
        def find_nearest(obj, k):
            return list(obj.values())[np.argmin(np.abs(np.array(list(obj.keys())) - k))]

        lower_x_for_y = {  # [m]
            -0.2: 0.29,
            -0.15: 0.31,
            -0.10: 0.32,
            -0.05: 0.35,
            0.0: 0.37,
            0.05: 0.38,
            0.1: 0.38,
            0.15: 0.35,
        }
        upper_x_for_y = {  # [m]
            -0.2: 0.64,
            0.0: 0.68,
            0.2: 0.64,
        }
        lower_x = find_nearest(lower_x_for_y, camera_frame.y)
        upper_x = find_nearest(upper_x_for_y, camera_frame.y)
        return (lower_x < camera_frame.x < upper_x) and (0.28 < camera_frame.z < 0.6)  # [m]
