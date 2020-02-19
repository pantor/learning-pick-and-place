from typing import List

from loguru import logger
import numpy as np
import cv2

from actions.action import Action, RobotPose
from orthographical import OrthographicImage
from cfrankr import Affine
from utils.image import crop, get_area_of_interest_new


class Converter:
    def __init__(self, grasp_z_offset=0.0, shift_z_offset=0.0, place_z_offset=0.0, box=None):
        self.action_should_be_safe = True
        self.grasp_z_offset = grasp_z_offset
        self.shift_z_offset = shift_z_offset
        self.place_z_offset = place_z_offset

        if not box:
            raise Exception('Could not find box inforation.')

        self.box_center = box['center']
        self.box_size = box['size']

    def is_pose_inside_box(self, pose: RobotPose, offset=0.0) -> bool:
        gripper_one_side_size = 0.5 * (pose.d + 0.002)  # [m]
        gripper_b1 = (pose * Affine(y=gripper_one_side_size)).translation()[0:2]
        gripper_b2 = (pose * Affine(y=-gripper_one_side_size)).translation()[0:2]

        rect = (
            (np.array(self.box_center) - (np.array(self.box_size) + np.array([offset, offset, 0])) / 2)[0:2],
            (np.array(self.box_center) + (np.array(self.box_size) - np.array([offset, offset, 0])) / 2)[0:2],
        )

        jaw1_inside_box = (rect[0] < gripper_b1).all() and (gripper_b1 < rect[1]).all()
        jaw2_inside_box = (rect[0] < gripper_b2).all() and (gripper_b2 < rect[1]).all()

        if (pose.b != 0.0 or pose.c != 0.0) and np.isfinite(pose.z):
            # start_point = (Affine(z=0.14) * pose.translation()
            # start_point_inside_box = (rect[0] < start_point).all() and (start_point < rect[1]).all()
            start_point_inside_box = True
        else:
            start_point_inside_box = True

        return jaw1_inside_box and jaw2_inside_box and start_point_inside_box

    def calculate_pose(self, action: Action, images: List[OrthographicImage]) -> None:
        if action.type == 'grasp':
            self.grasp_convert(action, images)
            is_safe = self.grasp_check_safety(action, images)

        elif action.type == 'shift':
            self.shift_convert(action, images)
            is_safe = self.shift_check_safety(action, images)

        elif action.type == 'place':
            self.place_convert(action, images)
            is_safe = self.place_check_safety(action, images)

        else:
            raise Exception('Unknown action type f{action.type}.')

        if not self.action_should_be_safe:
            is_safe = True

        if not is_safe:
            action.safe = -1
        elif not np.isfinite(action.pose.translation()).all():
            action.safe = 0
        else:
            action.safe = 1

    def grasp_check_safety(self, action: Action, images: List[OrthographicImage]) -> bool:
        return self.is_pose_inside_box(action.pose)

    def grasp_convert(self, action: Action, images: List[OrthographicImage]) -> None:
        image = images[0]
        mat_area_image = get_area_of_interest_new(image, action.pose, border_color=np.nan, project_3d=False, flags=cv2.INTER_NEAREST).mat

        mat_area_image = mat_area_image.astype(np.float32)
        mat_area_image[mat_area_image < 10 * 255] = np.nan  # Make every not found pixel NaN

        # Get distance at gripper for possible collisions
        gripper_one_side_size = 0.5 * image.pixel_size * (action.pose.d + 0.002)  # [px]
        area_center = crop(mat_area_image, (image.pixel_size * 0.012, image.pixel_size * 0.012))
        side_gripper_image_size = (image.pixel_size * 0.025, image.pixel_size * 0.025)
        area_left = crop(mat_area_image, side_gripper_image_size, (-gripper_one_side_size, 0))
        area_right = crop(mat_area_image, side_gripper_image_size, (gripper_one_side_size, 0))

        # Z is positive!
        z_raw = image.depth_from_value(np.nanmedian(area_center))
        if z_raw is np.NaN:
            area_center = crop(mat_area_image, (image.pixel_size * 0.03, image.pixel_size * 0.03))
            z_raw = image.depth_from_value(np.nanmedian(area_center))

        z_raw_left = image.depth_from_value(np.nanmin(area_left))
        z_raw_right = image.depth_from_value(np.nanmin(area_right))

        z_raw += self.grasp_z_offset
        z_raw_collision = min(z_raw_left, z_raw_right) - 0.008  # [m]
        z_raw_max = min(z_raw, z_raw_collision)  # Get the maximum [m] for impedance mode
        action.pose.z = -z_raw_max

    def shift_check_safety(self, action: Action, images: List[OrthographicImage]) -> bool:
        start_pose_inside_box = self.is_pose_inside_box(action.pose)

        end_pose = RobotPose(affine=(action.pose * Affine(x=action.shift_motion[0]*0.2, y=action.shift_motion[1]*0.2)))
        end_pose.d = action.pose.d

        end_pose_inside_box = self.is_pose_inside_box(end_pose)
        return start_pose_inside_box and end_pose_inside_box

    def shift_convert(self, action: Action, images: List[OrthographicImage]) -> None:
        image = images[0]
        mat_area_image = get_area_of_interest_new(image, action.pose, border_color=np.nan).mat

        mat_area_image = mat_area_image.astype(np.float32)
        mat_area_image[mat_area_image == 0] = np.nan  # Make every not found pixel NaN

        # Get distance at gripper for possible collisions
        area_center = crop(mat_area_image, (image.pixel_size * 0.01, image.pixel_size * 0.03))

        z_raw = image.depth_from_value(np.nanmax(area_center))
        z_raw += self.shift_z_offset
        action.pose.z = -z_raw  # [m] Move slightly up for gripper center point

    def place_check_safety(self, action: Action, images: List[OrthographicImage]) -> bool:
        return self.is_pose_inside_box(action.pose)

    def place_convert(self, action: Action, images: List[OrthographicImage]) -> None:
        image = images[0]
        mat_area_image = get_area_of_interest_new(image, action.pose, border_color=np.nan).mat

        mat_area_image = mat_area_image.astype(np.float32)
        mat_area_image[mat_area_image == 0] = np.nan  # Make every not found pixel NaN

        # Get distance at gripper for possible collisions
        gripper_one_side_size = 0.5 * image.pixel_size * (action.pose.d + 0.002)  # [px]
        area_center = crop(mat_area_image, (image.pixel_size * 0.025, image.pixel_size * 0.025))
        side_gripper_image_size = (image.pixel_size * 0.025, image.pixel_size * 0.025)
        area_left = crop(mat_area_image, side_gripper_image_size, (-gripper_one_side_size, 0))
        area_right = crop(mat_area_image, side_gripper_image_size, (gripper_one_side_size, 0))

        z_raw = image.depth_from_value(np.nanmedian(area_center))
        z_raw_left = image.depth_from_value(np.nanmin(area_left))
        z_raw_right = image.depth_from_value(np.nanmin(area_right))

        if z_raw is np.NaN:
            area_center = crop(mat_area_image, (image.pixel_size * 0.022, image.pixel_size * 0.03))
            z_raw = image.depth_from_value(np.nanmedian(area_center))

        z_raw += self.place_z_offset
        z_raw_collision = min(z_raw_left, z_raw_right) - 0.008  # [m]
        # z_raw_max = min(z_raw, z_raw_collision)  # Get the maximum [m] for impedance mode
        z_raw_max = z_raw

        action.pose.z = -z_raw_max  # [m] Move slightly up for gripper center point
