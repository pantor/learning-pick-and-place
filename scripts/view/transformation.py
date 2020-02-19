from pathlib import Path
import time

import cv2

from config import Config
from data.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest_new


if __name__ == '__main__':
    lateral = False
    suffix = 'ed-lateral_b-0_400' if lateral else 'ed-v'
    action, image = Loader.get_action('placing-3', '2019-12-12-16-07-12-857', 0, 'ed-v')

    # image = image.translate((0.0, 0.0, 0.05))
    # image = image.rotate_x(-0.3, (0.0, 0.25))

    draw_around_box(image, box=Config.box)
    # draw_pose(image, action.pose, convert_to_rgb=True)

    size_input = image.mat.shape[::-1]
    size_cropped = (200, 200)
    size_result = (32, 32)

    scale = 4
    image.mat = cv2.resize(image.mat, (size_input[0] // scale, size_input[1] // scale))
    image.pixel_size /= scale

    s = time.time()

    area_image = get_area_of_interest_new(
        image,
        action.pose,
        size_cropped=(size_cropped[0] // scale, size_cropped[1] // scale),
        size_result=size_result,
        border_color=70
    )

    print(time.time() - s)

    # area_image = get_area_of_interest_new(
    #     image,
    #     action.pose,
    #     size_cropped=size_cropped,
    #     size_result=size_result,
    #     border_color=70
    # )

    cv2.imwrite(str(Path.home() / 'Documents' / 'bin_picking' / 'test.png'), area_image.mat)

    # cv2.imshow('image', image.mat)
    # cv2.imshow('area image', area_image.mat)
    # cv2.waitKey(4000)
