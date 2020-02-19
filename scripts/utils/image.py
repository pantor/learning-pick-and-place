from typing import Dict, List, Tuple

import cv2
import numpy as np

from actions.action import RobotPose
from orthographical import OrthographicImage
from cfrankr import Affine
from utils.frames import Frames


def clone(image: OrthographicImage) -> OrthographicImage:
    return OrthographicImage(
        image.mat.copy(),
        image.pixel_size,
        image.min_depth,
        image.max_depth,
        image.camera,
        image.pose,
    )


def crop(mat_image: np.ndarray, size_output: Tuple[float, float], vec=(0, 0)) -> np.ndarray:
    margin_x_lower = int(round((mat_image.shape[0] - size_output[0]) / 2 + vec[1]))
    margin_y_lower = int(round((mat_image.shape[1] - size_output[1]) / 2 + vec[0]))
    margin_x_upper = margin_x_lower + int(round(size_output[0]))
    margin_y_upper = margin_y_lower + int(round(size_output[1]))
    return mat_image[margin_x_lower:margin_x_upper, margin_y_lower:margin_y_upper]


def get_transformation(x: float, y: float, a: float, center: Tuple[float, float], scale = 1.0, cropped: Tuple[float, float] = None):  # [rad]
    rot_mat = cv2.getRotationMatrix2D((round(center[0] - x), round(center[1] - y)), a * 180.0 / np.pi, scale)  # [deg]
    rot_mat[0][2] += x + scale * cropped[0] / 2 - center[0] if cropped else x
    rot_mat[1][2] += y + scale * cropped[1] / 2 - center[1] if cropped else y
    return rot_mat


# def get_area_of_interest(
#         image: OrthographicImage,
#         pose: Affine,
#         size_cropped: Tuple[float, float] = None,
#         size_result: Tuple[float, float] = None,
#         border_color=None,
#         project_3d=True,
# ) -> OrthographicImage:
#     size_input = (image.mat.shape[1], image.mat.shape[0])
#     center_image = (size_input[0] / 2, size_input[1] / 2)

#     action_pose = Frames.get_pose_in_image(action_pose=pose, image_pose=Affine(image.pose)) if project_3d else pose

#     angle_a = action_pose.a
#     if abs(action_pose.b) > np.pi - 0.1 and abs(action_pose.c) > np.pi - 0.1:
#         angle_a = action_pose.a - np.pi

#     trans = get_transformation(
#         image.pixel_size * action_pose.y,
#         image.pixel_size * action_pose.x,
#         -angle_a,
#         center_image
#     )
#     mat_result = cv2.warpAffine(image.mat, trans, size_input, borderValue=border_color)

#     new_pixel_size = image.pixel_size

#     if size_cropped:
#         mat_result = crop(mat_result, size_output=size_cropped)

#     if size_result:
#         mat_result = cv2.resize(mat_result, size_result)
#         new_pixel_size *= size_result[0] / size_cropped[0] if size_cropped else size_result[0] / size_input[0]

#     return OrthographicImage(
#         mat_result,
#         new_pixel_size,
#         image.min_depth,
#         image.max_depth,
#         image.camera,
#         Affine(x=action_pose.x, y=action_pose.y, a=-action_pose.a) * Affine(image.pose),
#     )


def get_area_of_interest_new(
        image: OrthographicImage,
        pose: Affine,
        size_cropped: Tuple[float, float] = None,
        size_result: Tuple[float, float] = None,
        border_color=None,
        project_3d=True,
        flags=cv2.INTER_LINEAR,
) -> OrthographicImage:
    size_input = (image.mat.shape[1], image.mat.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)

    action_pose = Frames.get_pose_in_image(action_pose=pose, image_pose=Affine(image.pose)) if project_3d else pose

    angle_a = action_pose.a
    if abs(action_pose.b) > np.pi - 0.1 and abs(action_pose.c) > np.pi - 0.1:
        angle_a = action_pose.a - np.pi

    if size_result and size_cropped:
        scale = size_result[0] / size_cropped[0]
        assert scale == (size_result[1] / size_cropped[1])
    elif size_result:
        scale = size_result[0] / size_input[0]
        assert scale == (size_result[1] / size_input[1])
    else:
        scale = 1.0

    size_final = size_result or size_cropped or size_input

    trans = get_transformation(
        image.pixel_size * action_pose.y,
        image.pixel_size * action_pose.x,
        -angle_a,
        center_image,
        scale=scale,
        cropped=size_cropped,
    )
    mat_result = cv2.warpAffine(image.mat, trans, size_final, flags=flags, borderValue=border_color)  # INTERPOLATION_METHOD

    image_pose = Affine(x=action_pose.x, y=action_pose.y, a=-action_pose.a) * Affine(image.pose)

    return OrthographicImage(
        mat_result,
        scale * image.pixel_size,
        image.min_depth,
        image.max_depth,
        image.camera,
        image_pose.to_array(),
    )


def patch_image_at(
        image: OrthographicImage,
        patch: np.ndarray,
        pose: Affine,
        size_cropped=None,
        operation='replace',
    ) -> OrthographicImage:
    if size_cropped:
        patch = cv2.resize(patch, size_cropped)

    size_input = (image.mat.shape[1], image.mat.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)
    center_cropped_image = (patch.shape[1] / 2, patch.shape[0] / 2)

    dx = center_image[0] - center_cropped_image[0] - image.pixel_size * pose.y
    dy = center_image[1] - center_cropped_image[1] - image.pixel_size * pose.x
    trans = get_transformation(
        dx * np.cos(pose.a) - dy * np.sin(pose.a),
        dy * np.cos(pose.a) + dx * np.sin(pose.a),
        pose.a,
        center_cropped_image
    )

    result = cv2.warpAffine(patch, trans, size_input, borderValue=np.iinfo(np.uint16).max)
    mask = np.array(result < np.iinfo(np.uint16).max, dtype=np.uint16)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint16), iterations=1)

    if operation == 'replace':
        mat_patched_image = np.where(mask, result, image.mat)
    elif operation == 'add':
        mat_patched_image = image.mat + np.where(mask, result, np.zeros(image.mat.shape, dtype=np.uint16))
    else:
        raise Exception(f'Operation {operation} not implemented in patch image!')

    return OrthographicImage(
        mat_patched_image,
        image.pixel_size,
        image.min_depth,
        image.max_depth,
        image.camera,
        image.pose,
    )


def get_rect(size: List[float], center=np.array([0.0, 0.0, 0.0])) -> List[Affine]:
    return [
        Affine(center[0] + size[0] / 2, center[1] + size[1] / 2, size[2]),
        Affine(center[0] + size[0] / 2, center[1] - size[1] / 2, size[2]),
        Affine(center[0] - size[0] / 2, center[1] - size[1] / 2, size[2]),
        Affine(center[0] - size[0] / 2, center[1] + size[1] / 2, size[2]),
    ]


def get_distance_to_box(image: OrthographicImage, box: Dict[str, List[float]]) -> float:
    bin_border = get_rect(size=box['size'], center=box['center'])
    image_pose = Affine(image.pose)
    return min((image_pose * border.inverse()).z for border in bin_border)


def draw_around_box(image: OrthographicImage, box: Dict[str, List[float]], draw_lines=False) -> None:
    bin_border = get_rect(size=box['size'], center=box['center'])
    image_border = get_rect(size=[10.0, 10.0, box['size'][2]])

    image_pose = Affine(image.pose)
    color_divisor = 255 * 255 if image.mat.dtype == np.float32 else 1

    bin_border_projection = [image.project((image_pose * p).to_array()) for p in bin_border]

    if draw_lines:
        color_direction = np.array([0, 255 * 255, 0]) / color_divisor  # Green

        for i in range(len(bin_border)):
            cv2.line(
                image.mat, tuple(bin_border_projection[i]),
                tuple(bin_border_projection[(i + 1) % len(bin_border)]),
                color_direction, 1
            )
    else:
        color = np.array([
            max(image.value_from_depth((image_pose * border.inverse()).z) for border in bin_border)
        ] * 3) / color_divisor

        image_border_projection = [image.project((image_pose * p).to_array()) for p in image_border]
        cv2.fillPoly(image.mat, np.array([image_border_projection, bin_border_projection]), color)


def draw_line(
        image: OrthographicImage,
        action_pose: Affine,
        pt1: Affine,
        pt2: Affine,
        color,
        thickness=1,
        reference_pose=None,
    ) -> None:
    action_pose = Frames.get_pose_in_image(action_pose=action_pose, image_pose=Affine(image.pose), reference_pose=reference_pose)
    pt1_projection = tuple(image.project((action_pose * pt1).to_array()))
    pt2_projection = tuple(image.project((action_pose * pt2).to_array()))
    cv2.line(image.mat, pt1_projection, pt2_projection, color, thickness, lineType=cv2.LINE_AA)


def draw_pose(image: OrthographicImage, action_pose: RobotPose, convert_to_rgb=False, reference_pose=None) -> None:
    if convert_to_rgb and image.mat.ndim == 2:
        image.mat = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)

    color_rect = (255 * 255, 0, 0)  # Blue
    color_lines = (0, 0, 255 * 255)  # Red
    color_direction = (0, 255 * 255, 0)  # Green

    rect = get_rect([200.0 / image.pixel_size, 200.0 / image.pixel_size, 0.0])
    for i, r in enumerate(rect):
        draw_line(image, action_pose, r, rect[(i + 1) % len(rect)], color_rect, 2, reference_pose)

    draw_line(image, action_pose, Affine(90 / image.pixel_size, 0), Affine(100 / image.pixel_size, 0), color_rect, 2, reference_pose)
    draw_line(image, action_pose, Affine(0.012, action_pose.d / 2), Affine(-0.012, action_pose.d / 2), color_lines, 1, reference_pose)
    draw_line(image, action_pose, Affine(0.012, -action_pose.d / 2), Affine(-0.012, -action_pose.d / 2), color_lines, 1, reference_pose)
    draw_line(image, action_pose, Affine(0, action_pose.d / 2), Affine(0, -action_pose.d / 2), color_lines, 1, reference_pose)
    draw_line(image, action_pose, Affine(0.006, 0), Affine(-0.006, 0), color_lines, 1, reference_pose)

    if not isinstance(action_pose.z, str) and np.isfinite(action_pose.z):
        draw_line(image, action_pose, Affine(z=0.14), Affine(), color_direction, 1, reference_pose)


def image_difference(image: OrthographicImage, image_comp: OrthographicImage) -> OrthographicImage:
    kernel = np.ones((5, 5), np.uint8)
    diff = np.zeros(image.mat.shape, np.uint8)
    diff[(image.mat > image_comp.mat + 5) & (image.mat > 0)] = 255
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=2)

    return OrthographicImage(
        diff,
        image.pixel_size,
        image.min_depth,
        image.max_depth,
        image.camera,
        image.pose,
    )


def get_zero_percentage(image: OrthographicImage) -> float:
    return 1 - cv2.countNonZero(image.mat) / image.mat.size
