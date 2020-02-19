import argparse
import time

import cv2

from actions.converter import Converter
from config import Config
from data.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sync robot learning collections via rsync.')
    parser.add_argument('-c', '--collection', dest='collection', type=str, required=True)
    parser.add_argument('-e', '--episode', dest='episode', type=str, required=True)
    parser.add_argument('-a', '--action', dest='action', type=int, default=0)
    parser.add_argument('-s', '--suffix', dest='suffix', type=str, default='ed-v')
    parser.add_argument('-m', '--model', dest='model', type=str)
    parser.add_argument('-d', '--draw', action='store_true')
    parser.add_argument('--area', action='store_true')
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    action, image = Loader.get_action(args.collection, args.episode, args.action, args.suffix)

    print('Action: ', action)

    if args.area:
        area_image = get_area_of_interest_new(image, action.pose, size_cropped=(200, 200))

        if args.convert:
            converter = Converter(grasp_z_offset=0.015, box=Config.box)
            converter.grasp_convert(action, [image])

        if args.show:
            cv2.imshow('area_image', area_image.mat)

    else:
        if args.draw:
            draw_around_box(image, box=Config.box)
            draw_pose(image, action.pose, convert_to_rgb=True)

        if args.show:
            cv2.imshow('image', image.mat)

    if args.show:
        cv2.waitKey(3000)


