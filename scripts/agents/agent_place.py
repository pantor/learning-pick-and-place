from typing import List

from loguru import logger
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=E0401

from actions.action import Action
from actions.converter import Converter
from actions.indexer import GraspIndexer
from config import Config
from data.loader import Loader
from inference.planar import InferencePlanarPose
from inference.planar_place import InferencePlanarPlace
from orthographical import OrthographicImage
from utils.epoch import Epoch
from utils.param import SelectionMethod


class Agent:
    def __init__(self, use_goal_images, **params):
        self.use_goal_images = use_goal_images
        self.network_type = Config.place_model[-1] if isinstance(Config.place_model, str) else Config.place_model[1][-1]

        if self.use_goal_images:
            self.model = Config.place_model
            self.watch_for_model_modification = True
            self.model_last_modified = Loader.get_model_path(self.model).stat().st_mtime

            number_top_grasp = 200  # default=200
            number_top_place = 200  # default=200

            combined_model = Loader.get_model(self.model)
            g = combined_model.get_layer('grasp')
            p = combined_model.get_layer('place')
            m = combined_model.get_layer('merge')

            # self.grasp_output = ['reward_grasp', 'z_m0', 'z_m1', 'z_m2']
            self.grasp_output = ['reward_grasp', 'z_m0']
            # self.grasp_output = ['reward_grasp', 'z_m']
            self.place_output = ['reward_place', 'z_p']
            self.merge_output = ['reward_merge']

            # Merge model repeat & tile
            z_g = tf.keras.Input(shape=m.input[0].shape[1:], name='z_g')
            z_p = tf.keras.Input(shape=m.input[1].shape[1:], name='z_p')

            z_g_repeated = tf.keras.backend.repeat_elements(z_g, number_top_place, axis=0)
            z_p_repeated = tf.tile(z_p, (number_top_grasp,) + tuple(1 for _ in z_g.shape[1:]))

            reward_merged = m([z_g_repeated, z_p_repeated])
            reward_merged_reshaped = tf.keras.backend.reshape(reward_merged, (number_top_grasp, number_top_place, 1))


            self.place_inference = InferencePlanarPlace(
                network_type=self.network_type,
                grasp_model=Model(inputs=g.input, outputs=[g.get_layer(l).output for l in self.grasp_output]),
                place_model=Model(inputs=p.input, outputs=[p.get_layer(l).output for l in self.place_output]),
                merge_model=Model(inputs=[z_g, z_p], outputs=reward_merged_reshaped),
                number_top_grasp=number_top_grasp,
                number_top_place=number_top_place,
                box=Config.box,
                lower_random_pose=Config.lower_random_pose,
                upper_random_pose=Config.upper_random_pose,
            )
            # self.place_inference.keep_indixes = [0]

        else:
            # combined_model = Loader.get_model(Config.place_model)
            # g = combined_model.get_layer('grasp')

            # self.grasp_inference = InferencePlanarPose(
            #     model=Model(inputs=g.input, outputs=g.get_layer('reward_grasp').output),
            #     box=Config.box,
            #     lower_random_pose=Config.lower_random_pose,
            #     upper_random_pose=Config.upper_random_pose,
            # )

            self.grasp_inference = InferencePlanarPose(
                model=Loader.get_model(Config.grasp_model, output_layer='prob'),
                box=Config.box,
                lower_random_pose=Config.lower_random_pose,
                upper_random_pose=Config.upper_random_pose,
            )

        self.indexer = GraspIndexer(gripper_classes=Config.gripper_classes)
        self.converter = Converter(
            grasp_z_offset=Config.grasp_z_offset,
            place_z_offset=Config.place_z_offset,
            box=Config.box
        )

        self.reinfer_next_time = True  # Always true in contrast to AgentPredict

        self.successful_grasp_before = False

    def check_for_model_reload(self):
        current_model_st_mtime = Loader.get_model_path(self.model).stat().st_mtime
        if self.watch_for_model_modification and current_model_st_mtime > self.model_last_modified + 0.5:  # [s]
            logger.warning(f'Reload model {self.model}.')
            try:
                combined_model = Loader.get_model(self.model)
                g = combined_model.get_layer('grasp')
                p = combined_model.get_layer('place')
                m = combined_model.get_layer('merge')

                self.place_inference.grasp_model = Model(inputs=g.input, outputs=[g.get_layer(l).output for l in self.grasp_output])
                self.place_inference.place_model = Model(inputs=p.input, outputs=[p.get_layer(l).output for l in self.place_output])
                self.place_inference.merge_model = Model(inputs=m.input, outputs=[m.get_layer(l).output for l in self.merge_output])
                self.model_last_modified = Loader.get_model_path(self.model).stat().st_mtime
            except OSError:
                logger.info('Could not load model, probabily file locked.')

    def infer(
            self,
            images: List[OrthographicImage],
            method: SelectionMethod,
            goal_images: List[OrthographicImage],
            place_images: List[OrthographicImage] = None,
            **params,
        ) -> List[Action]:

        if self.use_goal_images:
            if not goal_images:
                raise Exception('No goal images specified!')

            self.check_for_model_reload()

            if self.network_type == '1':
                grasp, place = self.place_inference.infer(images, goal_images, method, place_images)
            elif self.network_type == '2':
                grasp, place = self.place_inference.infer(images, goal_images, method, place_images=place_images)
        else:
            place_method = method if method in [SelectionMethod.RandomInference] else SelectionMethod.Random

            grasp = self.grasp_inference.infer(images, method)
            place = self.grasp_inference.infer(images, place_method)
            place.type = 'place'

        self.indexer.to_action(grasp)

        estimated_reward_lower_than_threshold = grasp.estimated_reward < Config.bin_empty_at_max_probability
        bin_empty = estimated_reward_lower_than_threshold and Epoch.selection_method_should_be_high(method)

        if bin_empty:
            return [Action('bin_empty', safe=1)]

        self.converter.calculate_pose(grasp, images)
        if place_images:
            self.converter.calculate_pose(place, place_images)
        else:
            self.converter.calculate_pose(place, images)

        return [grasp, place]

    def reward_for_action(self, images: List[OrthographicImage], action: Action) -> float:
        estimated_rewards = self.grasp_inference.infer_at_pose(images, action.pose)
        if isinstance(estimated_rewards, tuple):
            estimated_rewards, _ = estimated_rewards

        index = self.indexer.from_action(action)
        return estimated_rewards[0][0][index]
