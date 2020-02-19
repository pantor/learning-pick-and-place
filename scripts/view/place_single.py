import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras import Model  # pylint: disable=E0401

from actions.action import Affine, RobotPose
from config import Config
from data.loader import Loader
from utils.image import draw_around_box, get_area_of_interest_new


if __name__ == '__main__':
    save_path = Path(__file__).parent.parent.parent / 'test' / 'generated'

    collection = 'placing-3'
    episode_id = '2020-01-30-11-30-51-981'

    combined_model = Loader.get_model('placing-3-21-part-type-2')

    action_grasp, image_grasp = Loader.get_action(collection, episode_id, 0, 'ed-v')
    action_place, image_place, image_goal = Loader.get_action(collection, episode_id, 1, ['ed-v', 'ed-goal'])

    draw_around_box(image_grasp, box=Config.box)
    draw_around_box(image_place, box=Config.box)
    draw_around_box(image_goal, box=Config.box)

    pose_grasp = action_grasp.pose
    # pose_grasp = RobotPose(Affine(x=-0.0053, y=0.0414, a=1.4708))

    pose_place = action_place.pose
    # pose_place = RobotPose(Affine(x=-0.0025, y=0.0563, a=-1.4708))

    image_grasp_area = get_area_of_interest_new(image_grasp, pose_grasp, size_cropped=(200, 200), size_result=(32, 32)).mat
    image_place_area = get_area_of_interest_new(image_place, pose_place, size_cropped=(200, 200), size_result=(32, 32)).mat
    image_goal_area = get_area_of_interest_new(image_goal, pose_place, size_cropped=(200, 200), size_result=(32, 32)).mat

    g = combined_model.get_layer('grasp')
    p = combined_model.get_layer('place')
    m = combined_model.get_layer('merge')

    grasp_output = ['reward_grasp', 'z_m0']
    place_output = ['reward_place', 'z_p']
    merge_output = ['reward_merge']

    grasp_model = Model(inputs=g.input, outputs=[g.get_layer(l).output for l in grasp_output])
    place_model = Model(inputs=p.input, outputs=[p.get_layer(l).output for l in place_output])
    merge_model = Model(inputs=m.input, outputs=[m.get_layer(l).output for l in merge_output])

    input_grasp = np.expand_dims(image_grasp_area, axis=3) / (255 * 255)
    input_place = np.expand_dims(image_place_area, axis=3) / (255 * 255)
    input_goal = np.expand_dims(image_goal_area, axis=3) / (255 * 255)

    result_m, *z_m_list = grasp_model.predict([[input_grasp]])
    result_p, z_p = place_model.predict(([input_place], [input_goal]))
    result_merged = [merge_model.predict((z_m_i[0][0], z_p[0][0])) for z_m_i in z_m_list]

    print(pose_grasp)
    print(pose_place)

    print(result_m)
    print(result_p)
    print(result_merged)
    cv2.imwrite(str(save_path / 'test-grasp2.png'), image_grasp_area)
    cv2.imwrite(str(save_path / 'test-place.png'), image_place_area)
    cv2.imwrite(str(save_path / 'test-goal.png'), image_goal_area)
