import cv2
import numpy as np

from agents.agent_place import Agent
from config import Config
from data.loader import Loader
from utils.image import draw_pose
from utils.param import SelectionMethod


if __name__ == '__main__':
    agent_place = Agent(use_goal_images=True)

    if not Config.use_goal_images:
        raise Exception(f'Does not use goal images!')

    collection = 'placing-3'
    episode_id = '2020-02-04-00-34-54-455'

    if agent_place.network_type == '1':
        image = Loader.get_image(collection, episode_id, 0, 'ed-v')
        image_goal = Loader.get_image(collection, episode_id, 0, 'ed-goal')

        actions = agent_place.infer([image], SelectionMethod.Max, [image_goal])

    elif agent_place.network_type == '2':
        image = Loader.get_image(collection, episode_id, 0, 'ed-v')
        image_place = Loader.get_image(collection, episode_id, 1, 'ed-v')
        image_goal = Loader.get_image(collection, episode_id, 1, 'ed-goal')

        actions = agent_place.infer([image], SelectionMethod.Max, [image_goal], [image_place])
        actions = agent_place.infer([image], SelectionMethod.Max, [image_goal], [image_place])

    print(f'Grasp: {actions[0].estimated_reward:0.3f} Place: {actions[1].estimated_reward:0.3f}')

    print(actions[0].pose)
    print(actions[1].pose)

    draw_pose(image, actions[0].pose, convert_to_rgb=True)
    draw_pose(image_goal, actions[1].pose, convert_to_rgb=True)

    output_path = '/home/berscheid/Documents/bin_picking/test/generated/'

    cv2.imwrite(output_path + 'test-grasp.png', image.mat)
    cv2.imwrite(output_path + 'test-place.png', image_goal.mat)
