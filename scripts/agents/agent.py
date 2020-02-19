from typing import List

from loguru import logger

from actions.action import Action
from actions.converter import Converter
from actions.indexer import GraspIndexer, LateralIndexer, GraspFinalDIndexer
from config import Config
from data.loader import Loader
from inference.planar import InferencePlanarPose
from orthographical import OrthographicImage
from utils.epoch import Epoch
from utils.param import SelectionMethod


class Agent:
    def __init__(self, **params):
        self.model = Config.grasp_model
        self.watch_for_model_modification = True
        self.model_last_modified = Loader.get_model_path(self.model).stat().st_mtime

        self.monte_carlo = 40 if 'mc' in self.model[1] else None
        self.with_types = 'types' in self.model[1]

        self.output_layer = 'prob' if not self.with_types else ['prob', 'type']
        self.inference = InferencePlanarPose(
            model=Loader.get_model(self.model, output_layer=self.output_layer),
            box=Config.box,
            lower_random_pose=Config.lower_random_pose,
            upper_random_pose=Config.upper_random_pose,
            monte_carlo=self.monte_carlo,
            with_types=self.with_types,
        )
        self.inference.keep_indixes = None
        self.indexer = GraspIndexer(gripper_classes=Config.gripper_classes)
        self.converter = Converter(grasp_z_offset=Config.grasp_z_offset, box=Config.box)

        # # self.indexer = GraspFinalDIndexer(gripper_classes=Config.gripper_classes, final_d_classes=[0.0, 0.035])
        # self.indexer = LateralIndexer(
        #     angles=[(0, 0), (0.3, 0)],
        #     gripper_classes=[0.05, 0.07, 0.084],
        # )
        # self.converter = Converter(grasp_z_offset=Config.grasp_z_offset, box=Config.box)

        self.reinfer_next_time = True  # Always true in contrast to AgentPredict

    def check_for_model_reload(self):
        current_model_st_mtime = Loader.get_model_path(self.model).stat().st_mtime
        if self.watch_for_model_modification and current_model_st_mtime > self.model_last_modified + 0.5:  # [s]
            logger.warning(f'Reload model {self.model}.')
            try:
                self.inference.model = Loader.get_model(self.model, output_layer=self.output_layer)
                self.model_last_modified = Loader.get_model_path(self.model).stat().st_mtime
            except OSError:
                logger.info('Could not load model, probabily file locked.')

    def infer(self, images: List[OrthographicImage], method: SelectionMethod, **params) -> List[Action]:
        if self.monte_carlo:  # Adapt monte carlo progress parameter s
            epoch_in_collection = Loader.get_episode_count(Config.collection)
            s_not_bounded = (epoch_in_collection - 3500) * 1 / (4500 - 3500)
            self.inference.current_s = max(min(s_not_bounded, 1.0), 0.0)

        self.check_for_model_reload()

        if len(images) == 3:
            images[2].mat = images[2].mat[:, :, ::-1]  # BGR to RGB

        action = self.inference.infer(images, method)
        self.indexer.to_action(action)

        print(action, method)

        estimated_reward_lower_than_threshold = action.estimated_reward < Config.bin_empty_at_max_probability
        bin_empty = estimated_reward_lower_than_threshold and Epoch.selection_method_should_be_high(method)

        if bin_empty:
            return [Action('bin_empty', safe=1)]

        self.converter.calculate_pose(action, images)
        return [action]

    def reward_for_action(self, images: List[OrthographicImage], action: Action) -> float:
        estimated_rewards = self.inference.infer_at_pose(images, action.pose)
        if isinstance(estimated_rewards, tuple):
            estimated_rewards, _ = estimated_rewards

        index = self.indexer.from_action(action)
        return estimated_rewards[0][0][index]
