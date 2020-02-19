import time
from typing import List

import cv2
from loguru import logger
import numpy as np

from actions.action import Action, RobotPose
from cfrankr import Affine
from orthographical import OrthographicImage
from inference.inference import Inference
from utils.param import SelectionMethod


class InferencePlanarPlace(Inference):
    def __init__(self, network_type, grasp_model, place_model, merge_model, number_top_grasp, number_top_place, box, lower_random_pose=None, upper_random_pose=None):
        super(InferencePlanarPlace, self).__init__(
            None,
            box,
            lower_random_pose=lower_random_pose,
            upper_random_pose=upper_random_pose
        )

        self.network_type = network_type
        self.grasp_model = grasp_model
        self.place_model = place_model
        self.merge_model = merge_model

        self.number_top_grasp = number_top_grasp
        self.number_top_place = number_top_place

        self.a_space = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 37)  # [rad] # Don't use a=0.0

    def infer(
            self,
            images: List[OrthographicImage],
            goal_images: List[OrthographicImage],
            method: SelectionMethod,
            verbose=1,
            place_images: List[OrthographicImage] = None,
        ) -> List[Action]:

        start = time.time()

        if method == SelectionMethod.Random:
            grasp_action = Action(action_type='grasp')
            grasp_action.index = np.random.choice(range(3))
            grasp_action.pose = RobotPose(affine=Affine(
                x=np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                y=np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                a=np.random.uniform(self.lower_random_pose[3], self.upper_random_pose[3]),  # [rad]
            ))
            grasp_action.estimated_reward = -1
            grasp_action.estimated_reward_std = 0.0
            grasp_action.method = method
            grasp_action.step = 0

            place_action = Action(action_type='place')
            place_action.index = np.random.choice(range(3))
            place_action.pose = RobotPose(affine=Affine(
                x=np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                y=np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                a=np.random.uniform(self.lower_random_pose[3], self.upper_random_pose[3]),  # [rad]
            ))
            place_action.estimated_reward = -1
            place_action.estimated_reward_std = 0.0
            place_action.method = method
            place_action.step = 0

            return [grasp_action, place_action]

        input_images = [self.get_images(i) for i in images]
        goal_input_images = [self.get_images(i) for i in goal_images]

        if self.network_type == '2' and not place_images:
            raise Exception(f'Place images are missing for network type {self.network_type}')
        elif place_images:
            place_input_images = [self.get_images(i) for i in place_images]

        grasp_input = input_images + goal_input_images if self.network_type == '1' else input_images
        place_input = input_images + goal_input_images if self.network_type == '1' else place_input_images + goal_input_images

        m_reward, m_z = self.grasp_model.predict(grasp_input, batch_size=128)
        # m_reward, *m_z_list = self.grasp_model.predict(grasp_input, batch_size=128)
        # m_z_list = tuple(np.expand_dims(m_zi, axis=3) for m_zi in m_z_list)
        # m_z = np.concatenate(m_z_list, axis=3)

        p_reward, p_z = self.place_model.predict(place_input, batch_size=128)

        if self.keep_indixes:
            self.keep_array_at_last_indixes(m_reward, self.keep_indixes)

        first_method = SelectionMethod.PowerProb if method in [SelectionMethod.Top5, SelectionMethod.Max] else method
        # first_method = SelectionMethod.Top5
        filter_lambda_n_grasp = self.get_filter_n(first_method, self.number_top_grasp)
        filter_lambda_n_place = self.get_filter_n(first_method, self.number_top_place)

        m_top_index = filter_lambda_n_grasp(m_reward)
        p_top_index = filter_lambda_n_place(p_reward)

        m_top_index_unraveled = np.transpose(np.asarray(np.unravel_index(m_top_index, m_reward.shape)))
        p_top_index_unraveled = np.transpose(np.asarray(np.unravel_index(p_top_index, p_reward.shape)))

        # print(m_top_index_unraveled.tolist())
        # print(p_top_index_unraveled.tolist())

        m_top_z = m_z[m_top_index_unraveled[:, 0], m_top_index_unraveled[:, 1], m_top_index_unraveled[:, 2]]
        # m_top_z = m_z[m_top_index_unraveled[:, 0], m_top_index_unraveled[:, 1], m_top_index_unraveled[:, 2], m_top_index_unraveled[:, 3]]
        p_top_z = p_z[p_top_index_unraveled[:, 0], p_top_index_unraveled[:, 1], p_top_index_unraveled[:, 2]]

        reward = self.merge_model.predict([m_top_z, p_top_z], batch_size=2**12)

        m_top_reward = m_reward[m_top_index_unraveled[:, 0], m_top_index_unraveled[:, 1], m_top_index_unraveled[:, 2], m_top_index_unraveled[:, 3]]
        # p_top_reward = p_reward[p_top_index_unraveled[:, 0], p_top_index_unraveled[:, 1], p_top_index_unraveled[:, 2]]
        m_top_reward_repeated = np.repeat(np.expand_dims(np.expand_dims(m_top_reward, axis=1), axis=1), self.number_top_place, axis=1)

        filter_measure = reward * m_top_reward_repeated

        filter_lambda = self.get_filter(method)
        index_raveled = filter_lambda(filter_measure)

        index_unraveled = np.unravel_index(index_raveled, reward.shape)
        m_index = m_top_index_unraveled[index_unraveled[0]]
        p_index = p_top_index_unraveled[index_unraveled[1]]

        grasp_action = Action(action_type='grasp')
        grasp_action.index = m_index[3]
        grasp_action.pose = self.pose_from_index(m_index, m_reward.shape, images[0])
        grasp_action.estimated_reward = m_reward[tuple(m_index)]
        grasp_action.method = method
        grasp_action.step = 0

        place_action = Action(action_type='place')
        place_action.index = p_index[3]
        place_action.pose = self.pose_from_index(p_index, p_reward.shape, images[0], resolution_factor=1.0)
        place_action.estimated_reward = reward[index_unraveled]  # reward[index_raveled, 0]  # p_reward[tuple(p_index)]
        place_action.method = method
        place_action.step = 0

        if verbose:
            logger.info(f'NN inference time [s]: {time.time() - start:.3}')
        return [grasp_action, place_action]
