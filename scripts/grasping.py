#!/usr/bin/python3.6

import random
import time

from loguru import logger

import utils.path_fix_catkin  # pylint: disable=W0611
from utils.camera import Camera

from actions.action import RobotPose
from agents.agent import Agent
from agents.agent_place import Agent as AgentPlace
from agents.agent_predict import Agent as AgentPredict
from agents.agent_shift import Agent as AgentShift
from cfrankr import Affine, MotionData  # pylint: disable=E0611
from config import Config
from data.loader import Loader
from experiment import Experiment
from utils.episode import Episode
from utils.frames import Frames
from utils.goal_database import GoalDatabase
from utils.param import Mode


class BinPickingExperiment(Experiment):
    def __init__(self):
        super().__init__()

        # self.agent = Agent()
        self.agent = AgentPlace(use_goal_images=Config.use_goal_images)
        self.current_bin = Config.start_bin

    def take_goal_images(self, current_bin, current_goal_images):
        if current_goal_images and isinstance(current_goal_images[0], list):
            return None

        logger.info('Take goal images.')
        wait_input = input('<enter> to take single goal image, <p> to take multiple goal images, or <space> to keep current goal image.')

        if wait_input == '':
            image_frame = self.robot.current_pose(Frames.camera)
            goal_images = self.take_images(image_frame=image_frame, current_bin=current_bin)

        elif wait_input == 'p':
            goal_images = []

            while wait_input == 'p':
                image_frame = self.robot.current_pose(Frames.camera)
                goal_images.append(self.take_images(image_frame=image_frame, current_bin=current_bin))

                wait_input = input('Press <p> to append goal image or <enter> to continue.')

        elif wait_input.isdigit():
            image_frame = self.robot.current_pose(Frames.camera)
            goal_images_single = self.take_images(image_frame=image_frame, current_bin=current_bin)
            goal_images = [goal_images_single for _ in range(int(wait_input))]

        else:
            return None

        logger.info('Goal image taken.')
        input('Press Enter to continue...')

        return goal_images


    def manipulate(self) -> None:
        current_bin_episode = None
        goal_images = None

        for epoch in Config.epochs:
            while self.history.total() < epoch.number_episodes:
                current_episode = Episode()
                current_bin_episode = current_bin_episode if current_bin_episode else current_episode.id
                current_selection_method = self.get_current_selection_method(epoch)

                start = time.time()

                place_action_in_other_bin = Config.release_in_other_bin and not Config.release_during_grasp_action
                place_bin = Frames.get_next_bin(self.current_bin) if place_action_in_other_bin else self.current_bin

                if (not Config.predict_images) or self.agent.reinfer_next_time:
                    self.robot.recover_from_errors()

                    if not place_action_in_other_bin or Config.take_after_images:
                        self.robot.move_joints(Frames.bin_joint_values[self.current_bin], self.md)

                    b, c = random.choice(Config.overview_image_angles) if Config.lateral_overview_image else 0, 0
                    camera_frame_overview = Frames.get_camera_frame(self.current_bin, b=b, c=c)
                    if not Frames.is_camera_frame_safe(camera_frame_overview):
                        continue

                    if place_action_in_other_bin:
                        self.robot.move_cartesian(Frames.camera, Frames.get_camera_frame(place_bin, b=b, c=c), self.md)
                    elif Config.take_goal_images:
                        self.robot.move_cartesian(Frames.camera, camera_frame_overview, self.md)

                    if Config.take_goal_images:
                        new_goal_images = self.take_goal_images(current_bin=place_bin, current_goal_images=goal_images)
                        goal_images = new_goal_images if new_goal_images else goal_images

                    elif Config.use_goal_images:
                        attr = random.choice(GoalDatabase.get(Config.goal_images_dataset))
                        goal_images = [Loader.get_image(attr[0], attr[1], attr[2], s) for s in attr[3]]

                    if place_action_in_other_bin:
                        place_image_frame = self.robot.current_pose(Frames.camera)
                        place_images = self.take_images(image_frame=place_image_frame, current_bin=place_bin)

                    if Config.mode is Mode.Measure or Config.lateral_overview_image:
                        self.robot.move_cartesian(Frames.camera, camera_frame_overview, self.md)

                    image_frame = self.robot.current_pose(Frames.camera)
                    images = self.take_images(image_frame=image_frame)

                    if not Frames.is_gripper_frame_safe(self.robot.current_pose(Frames.gripper)):
                        logger.info('Image frame not safe!')
                        self.robot.recover_from_errors()
                        continue

                input_images = self.get_input_images(images)
                input_place_images = self.get_input_images(place_images) if place_action_in_other_bin else None
                input_goal_images = None

                if Config.use_goal_images:
                    if isinstance(goal_images, list) and isinstance(goal_images[0], list):
                        goal_images_single = goal_images.pop(0)
                    else:
                        goal_images_single = goal_images

                    input_goal_images = self.get_input_images(goal_images_single)

                actions = self.agent.infer(
                    input_images,
                    current_selection_method,
                    goal_images=input_goal_images,
                    place_images=input_place_images,
                )

                for action_id, action in enumerate(actions):
                    logger.info(f'Action ({action_id+1}/{len(actions)}): {action}')

                for action_id, action in enumerate(actions):
                    action.images = {}
                    action.save = True
                    action.bin = self.current_bin
                    action.bin_episode = current_bin_episode

                    current_action_place_in_other_bin = place_action_in_other_bin and action.type == 'place'
                    current_image_pose = place_image_frame if current_action_place_in_other_bin else image_frame
                    current_bin = place_bin if current_action_place_in_other_bin else self.current_bin

                    if Config.mode is Mode.Measure:
                        before_images = place_images if current_action_place_in_other_bin else images
                        self.saver.save_image(before_images, current_episode.id, action_id, 'v', action=action)

                        if Config.use_goal_images:
                            self.saver.save_image(goal_images_single, current_episode.id, action_id, 'goal', action=action)

                    self.saver.save_action_plan(action, current_episode.id)

                    logger.info(f'Executing action: {action_id} at time {time.time() - self.overall_start:0.1f}')

                    if Config.set_zero_reward:
                        action.safe = -1

                    execute_action = True

                    if action.type == 'bin_empty':
                        action.save = False
                        execute_action = False
                    elif action.type == 'new_image':
                        action.save = False
                        execute_action = False
                        self.agent.reinfer_next_time = True

                    if action.safe <= 0:
                        execute_action = False
                        action.collision = True

                        # Set actions after this action to unsafe
                        for a in actions[action_id + 1:]:
                            a.safe = action.safe

                        reason = 'not within box' if action.safe == -1 else 'not a number'
                        logger.warning(f'Action (type={action.type}) is {reason} (safe={action.safe}).')

                        if action.safe == 0 and action.type in ['grasp', 'shift']:
                            logger.warning(f'Episode is not saved.')
                            current_episode.save = False
                            break

                        if action.type == 'place' and action_id > 0:
                            prior_action = actions[action_id - 1]

                            if prior_action.type == 'grasp' and prior_action.reward > 0:
                                central_pose = RobotPose(affine=Affine(z=-0.28), d=action.pose.d)

                                action_frame = Frames.get_action_pose(action_pose=central_pose, image_pose=current_image_pose)
                                self.place(current_episode, action_id, action, action_frame, current_image_pose)

                    # Dont place if grasp was not successful
                    if action.type == 'place' and action_id > 0:
                        prior_action = actions[action_id - 1]

                        if prior_action.type == 'grasp' and (prior_action.reward == 0 or prior_action.safe < 1):
                            execute_action = False

                    if Config.take_lateral_images and action.save and Config.mode is Mode.Measure:
                        md_lateral = MotionData().with_dynamics(1.0)

                        for b, c in Config.lateral_images_angles:
                            lateral_frame = Frames.get_camera_frame(current_bin, a=action.pose.a, b=b, c=c, reference_pose=image_frame)

                            if not Frames.is_camera_frame_safe(lateral_frame) or (b == 0.0 and c == 0.0):
                                continue

                            lateral_move_succss = self.robot.move_cartesian(Frames.camera, lateral_frame, md_lateral)  # Remove a for global b, c pose
                            if lateral_move_succss:
                                self.saver.save_image(self.take_images(current_bin=current_bin), current_episode.id, action_id, f'lateral_b{b:0.3f}_c{c:0.3f}'.replace('.', '_'), action=action)

                    if execute_action:
                        action_frame = Frames.get_action_pose(action_pose=action.pose, image_pose=current_image_pose)

                        if Config.mode is Mode.Measure and Config.take_direct_images:
                            self.robot.move_cartesian(Frames.camera, Affine(z=0.308) * Affine(b=0.0, c=0.0) * action_frame)
                            self.saver.save_image(self.take_images(current_bin=current_bin), current_episode.id, action_id, 'direct', action=action)

                        if action.type == 'grasp':
                            self.grasp(current_episode, action_id, action, action_frame, current_image_pose)

                            if Config.use_goal_images and self.last_after_images and not place_action_in_other_bin:  # Next action is Place
                                place_action_id = action_id + 1
                                actions[place_action_id].pose.d = self.gripper.width()  # Use current gripper width for safety analysis
                                self.agent.converter.calculate_pose(actions[place_action_id], self.last_after_images)

                        elif action.type == 'shift':
                            old_reward_around_action = 0.0
                            self.shift(current_episode, action_id, action, action_frame, current_image_pose)
                            new_reward_around_action = 0.0

                            action.reward = new_reward_around_action - old_reward_around_action

                        elif action.type == 'place':
                            self.place(current_episode, action_id, action, action_frame, current_image_pose, place_bin=place_bin)
                            action.reward = actions[action_id - 1].reward

                    else:
                        if Config.take_after_images:
                            self.robot.move_cartesian(Frames.camera, current_image_pose, self.md)
                            self.saver.save_image(self.take_images(current_bin=current_bin), current_episode.id, action_id, 'after', action=action)

                    action.execution_time = time.time() - start
                    logger.info(f'Time for action: {action.execution_time:0.3f} [s]')

                    if action.save:
                        current_episode.actions.append(action)
                        self.history.append(current_episode)
                    else:
                        break

                    logger.info(f'Episodes (reward / done / total): {self.history.total_reward(action_type="grasp")} / {self.history.total()} / {sum(e.number_episodes for e in Config.epochs)}')
                    logger.info(f'Last success: {self.history.failed_grasps_since_last_success_in_bin(self.current_bin)} cycles ago.')

                    # history.save_grasp_rate_prediction_step_evaluation(Config.evaluation_path)

                # Switch bin
                should_change_bin_for_evaluation = (Config.mode is Mode.Evaluate and self.history.successful_grasps_in_bin(self.current_bin) == Config.change_bin_at_number_of_success_grasps)
                should_change_bin = (Config.mode is not Mode.Evaluate and (self.history.failed_grasps_since_last_success_in_bin(self.current_bin) >= Config.change_bin_at_number_of_failed_grasps or action.type == 'bin_empty'))
                if should_change_bin_for_evaluation or (Config.change_bins and should_change_bin):
                    if Config.mode is Mode.Evaluate:
                        pass
                        # history.save_grasp_rate_prediction_step_evaluation(Config.evaluation_path)

                    self.current_bin = Frames.get_next_bin(self.current_bin)
                    self.agent.reinfer_next_time = True
                    current_bin_episode = None
                    logger.info('Switch to other bin.')

                    if Config.mode is not Mode.Perform and Config.home_gripper:
                        self.gripper.homing()

                if Config.mode is Mode.Measure and current_episode.actions and current_episode.save:
                    logger.info(f'Save episode {current_episode.id}.')
                    self.saver.save_episode(current_episode)

                # Retrain model
                if Config.train_model and self.history.total() > 0 and not self.history.total() % Config.train_model_every_number_cycles:
                    logger.warning('Retrain model!')
                    self.retrain_model()

        logger.info('Finished cleanly.')


if __name__ == '__main__':
    exp = BinPickingExperiment()
    exp.init()
    exp.manipulate()
