#!/usr/bin/python3.6

from multiprocessing import Process
from subprocess import Popen
import sys
import time
from typing import List, Optional

from loguru import logger

import utils.path_fix_catkin  # pylint: disable=W0611
from utils.camera import Camera

from actions.action import Action, RobotPose
from cfrankr import Affine, Gripper, MotionData, Robot, Waypoint  # pylint: disable=E0611
from config import Config
from learning.utils.layers import one_hot_gen  # pylint: disable:unused-import
from utils.episode import Episode, EpisodeHistory
from utils.frames import Frames
from utils.saver import Saver
from utils.param import Mode, SelectionMethod
from orthographical import OrthographicImage



class Experiment:
    def __init__(self):
        self.camera = Camera(camera_suffixes=Config.camera_suffixes)
        self.history = EpisodeHistory()
        self.gripper = Gripper('172.16.0.2', Config.gripper_speed, Config.gripper_force)
        self.robot = Robot('panda_arm', Config.general_dynamics_rel)
        self.saver = Saver(Config.database_url, Config.collection)

        self.current_bin = Config.start_bin

        self.md = MotionData().with_dynamics(1.0)

        self.overall_start = 0

        self.last_after_images: Optional[List[OrthographicImage]] = None

    def move_to_release(self, md: MotionData, direct=False, target=None) -> bool:
        possible_random_affine = Affine()
        if Config.random_pose_before_release:
            possible_random_affine = Config.max_random_affine_before_release.get_inner_random()

        target = target if target else Frames.get_release_frame(Frames.get_next_bin(self.current_bin)) * possible_random_affine

        self.robot.recover_from_errors()

        if Config.mode is Mode.Measure:
            self.move_to_safety(md)

        if Config.release_in_other_bin:
            if Config.release_as_fast_as_possible:
                waypoints = [Waypoint(Frames.release_fastest, Waypoint.ReferenceType.ABSOLUTE)]
            else:
                waypoints = [Waypoint(target, Waypoint.ReferenceType.ABSOLUTE)]

                if not direct:
                    waypoints.insert(0, Waypoint(Frames.release_midway, Waypoint.ReferenceType.ABSOLUTE))

            return self.robot.move_waypoints_cartesian(Frames.gripper, waypoints, MotionData())

        return self.robot.move_cartesian(
            Frames.gripper,
            Frames.get_release_frame(self.current_bin) * possible_random_affine,
            MotionData()
        )

    def move_to_safety(self, md: MotionData) -> bool:
        move_up = max(0.0, 0.16 - self.robot.current_pose(Frames.gripper).z)
        return self.robot.move_relative_cartesian(Frames.gripper, Affine(z=move_up), md)

    def take_images(self, image_frame: Affine = None, current_bin=None) -> List[OrthographicImage]:
        current_bin = current_bin if current_bin else self.current_bin

        images = self.camera.take_images()
        if not image_frame:
            image_frame = self.robot.current_pose(Frames.camera)
        pose = RobotPose(affine=(image_frame.inverse() * Frames.get_frame(current_bin)))

        for image in images:
            image.pose = pose.to_array()

        return images

    def grasp(self, current_episode: Episode, action_id: int, action: Action, action_frame: Affine, image_frame: Affine):
        md_approach_down = MotionData().with_dynamics(0.3).with_z_force_condition(7.0)
        md_approach_up = MotionData().with_dynamics(1.0).with_z_force_condition(20.0)

        action_approch_affine = Affine(z=Config.approach_distance_from_pose)
        action_approach_frame = action_frame * action_approch_affine

        try:
            process_gripper = Process(target=self.gripper.move, args=(action.pose.d, ))
            process_gripper.start()

            self.robot.move_cartesian(Frames.gripper, action_approach_frame, self.md)

            process_gripper.join()
        except OSError:
            self.gripper.move(0.08)
            self.robot.move_cartesian(Frames.gripper, action_approach_frame, self.md)

        self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine.inverse(), md_approach_down)

        if md_approach_down.did_break:
            self.robot.recover_from_errors()
            action.collision = True
            self.robot.move_relative_cartesian(Frames.gripper, Affine(z=0.001), md_approach_up)

        action.final_pose = RobotPose(affine=(image_frame.inverse() * self.robot.current_pose(Frames.gripper)))

        first_grasp_successful = self.gripper.clamp()
        if first_grasp_successful:
            logger.info('Grasp successful at first.')
            self.robot.recover_from_errors()

            action_approch_affine = Affine(z=Config.approach_distance_from_pose)

            move_up_success = self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)
            if move_up_success and not md_approach_up.did_break:
                if Config.mode is Mode.Measure and Config.take_after_images and not Config.release_in_other_bin:
                    self.robot.move_cartesian(Frames.camera, image_frame, self.md)
                    self.last_after_images = self.take_images()
                    self.saver.save_image(self.last_after_images, current_episode.id, action_id, 'after', action=action)

                if Config.release_during_grasp_action:
                    move_to_release_success = self.move_to_release(self.md)
                    if move_to_release_success:
                        if self.gripper.is_grasping():
                            action.reward = 1.0
                            action.final_pose.d = self.gripper.width()

                        if Config.mode is Mode.Perform:
                            self.gripper.release(action.final_pose.d + 0.005)  # [m]
                        else:
                            self.gripper.release(action.pose.d + 0.005)  # [m]
                            self.move_to_safety(md_approach_up)

                        if Config.mode is Mode.Measure and Config.take_after_images and Config.release_in_other_bin:
                            self.robot.move_cartesian(Frames.camera, image_frame, self.md)
                            self.last_after_images = self.take_images()
                            self.saver.save_image(self.last_after_images, current_episode.id, action_id, 'after', action=action)
                else:
                    if Config.mode is not Mode.Perform:
                        self.move_to_safety(md_approach_up)

                    if self.gripper.is_grasping():
                        action.reward = 1.0
                        action.final_pose.d = self.gripper.width()

                    if Config.mode is Mode.Measure and Config.take_after_images:
                        self.robot.move_cartesian(Frames.camera, image_frame, self.md)
                        self.last_after_images = self.take_images()
                        self.saver.save_image(self.last_after_images, current_episode.id, action_id, 'after', action=action)

            else:
                self.gripper.release(action.pose.d + 0.002)  # [m]

                self.robot.recover_from_errors()
                self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)
                move_to_safety_success = self.move_to_safety(md_approach_up)
                if not move_to_safety_success:
                    self.robot.recover_from_errors()
                    self.robot.recover_from_errors()
                    self.move_to_safety(md_approach_up)

                self.gripper.move(self.gripper.max_width)

                self.move_to_safety(md_approach_up)
                if Config.mode is Mode.Measure and Config.take_after_images:
                    self.robot.move_cartesian(Frames.camera, image_frame, self.md)
                    self.last_after_images = self.take_images()
                    self.saver.save_image(self.last_after_images, current_episode.id, action_id, 'after', action=action)

        else:
            logger.info('Grasp not successful.')
            self.gripper.release(self.gripper.width() + 0.002)  # [m]

            self.robot.recover_from_errors()
            move_up_successful = self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)

            if md_approach_up.did_break or not move_up_successful:
                self.gripper.release(action.pose.d)  # [m]

                self.robot.recover_from_errors()
                self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)
                self.move_to_safety(md_approach_up)

            if Config.mode is Mode.Measure and Config.take_after_images:
                self.robot.move_cartesian(Frames.camera, image_frame, self.md)
                self.last_after_images = self.take_images()
                self.saver.save_image(self.last_after_images, current_episode.id, action_id, 'after', action=action)

    def shift(self, current_episode: Episode, action_id: int, action: Action, action_frame: Affine, image_frame: Affine):
        md_approach_down = MotionData().with_dynamics(0.15).with_z_force_condition(6.0)
        md_approach_up = MotionData().with_dynamics(0.6).with_z_force_condition(20.0)
        md_shift = MotionData().with_dynamics(0.1).with_xy_force_condition(10.0)

        action_approch_affine = Affine(z=Config.approach_distance_from_pose)
        action_approach_frame = action_approch_affine * action_frame

        try:
            process_gripper = Process(target=self.gripper.move, args=(action.pose.d, ))
            process_gripper.start()

            self.robot.move_cartesian(Frames.gripper, action_approach_frame, self.md)

            process_gripper.join()
        except OSError:
            self.gripper.move(0.08)
            self.robot.move_cartesian(Frames.gripper, action_approach_frame, self.md)

        self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine.inverse(), md_approach_down)

        if md_approach_down.did_break:
            self.robot.recover_from_errors()
            action.collision = True
            self.robot.move_relative_cartesian(Frames.gripper, Affine(z=0.001), md_approach_up)

        self.robot.move_relative_cartesian(Frames.gripper, Affine(x=action.shift_motion[0], y=action.shift_motion[1]), md_shift)
        self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine, md_approach_up)

        # Reward is set outside of this function, due to dependency on agent

    def place(self, current_episode: Episode, action_id: int, action: Action, action_frame: Affine, image_frame: Affine, place_bin=None):
        place_bin = place_bin if place_bin else self.current_bin

        self.move_to_safety(self.md)

        md_approach_down = MotionData().with_dynamics(0.22).with_z_force_condition(7.0)
        md_approach_up = MotionData().with_dynamics(1.0).with_z_force_condition(20.0)

        action_approch_affine = Affine(z=Config.approach_distance_from_pose)
        action_approach_frame = action_frame * action_approch_affine

        if Config.release_in_other_bin:
            self.move_to_release(self.md, target=action_approach_frame)
        else:
            self.robot.move_cartesian(Frames.gripper, action_approach_frame, self.md)

        self.robot.move_relative_cartesian(Frames.gripper, action_approch_affine.inverse(), md_approach_down)

        if md_approach_down.did_break:
            self.robot.recover_from_errors()
            action.collision = True
            self.robot.move_relative_cartesian(Frames.gripper, Affine(z=0.001), md_approach_up)

        action.final_pose = RobotPose(affine=(image_frame.inverse() * self.robot.current_pose(Frames.gripper)))

        action.pose.d = self.gripper.width()
        self.gripper.release(action.pose.d + 0.01)  # [m]

        if Config.mode is not Mode.Perform:
            self.move_to_safety(md_approach_up)

        if Config.mode is Mode.Measure and Config.take_after_images:
            self.robot.move_cartesian(Frames.camera, image_frame, self.md)
            self.last_after_images = self.take_images(current_bin=place_bin)
            self.saver.save_image(self.last_after_images, current_episode.id, action_id, 'after', action=action)

    def init(self) -> None:
        self.gripper.stop()

        self.robot.recover_from_errors()
        self.move_to_safety(self.md)
        move_joints_successful = self.robot.move_joints(Frames.bin_joint_values[self.current_bin], self.md)

        if not move_joints_successful:
            self.gripper.move(0.07)

            self.robot.recover_from_errors()
            self.move_to_safety(self.md)
            move_joints_successful = self.robot.move_joints(Frames.bin_joint_values[self.current_bin], self.md)

        if Config.mode is Mode.Measure and not Config.home_gripper:
            logger.warning('Want to measure without homing gripper?')
        elif Config.home_gripper:
            self.gripper.homing()

        self.move_to_safety(self.md)
        self.overall_start = time.time()

    def retrain_model(self) -> None:
        with open('/tmp/training.txt', 'wb') as out:
            p = Popen([sys.executable, str(Config.train_script)], stdout=out)
            if not Config.train_async:
                p.communicate()

    def get_current_selection_method(self, epoch) -> SelectionMethod:
        if Config.mode in [Mode.Evaluate, Mode.Perform]:
            return epoch.get_selection_method_perform(
                self.history.failed_grasps_since_last_success_in_bin(self.current_bin)
            )
        return epoch.get_selection_method()

    def get_input_images(self, images):
        return list(filter(lambda i: i.camera in Config.model_input_suffixes, images))

