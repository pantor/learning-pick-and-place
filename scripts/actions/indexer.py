from typing import Any, List, Optional

import numpy as np

from actions.action import Action
from cfrankr import Affine
from utils.frames import Frames


class Indexer:
    pose_threshold = 1e-3

    @classmethod
    def _index_of_nearest_element(cls, pos: float, array: List[float]) -> int:
        return np.argmin(np.abs(np.array(array) - pos))

    @classmethod
    def _index_of_nearest_pose(cls, pose: Affine, array: List[Any]) -> Optional[int]:
        array_diff = np.array([(pose.b - a[0])**2 + (pose.c - a[1])**2 for a in array])
        if np.min(array_diff) < cls.pose_threshold:
            raise Exception(f'Could not find a pose!')
        return np.argmin(array_diff)


class GraspIndexer(Indexer):
    def __init__(self, gripper_classes: List[float]):
        self.gripper_classes = gripper_classes

    def __len__(self) -> int:
        return len(self.gripper_classes)

    def to_action(self, action: Action) -> None:
        action.pose.d = self.gripper_classes[action.index]
        action.type = 'grasp'

    def from_action(self, action: Action, suffix='') -> int:
        return self._index_of_nearest_element(action.pose.d, self.gripper_classes)

    def from_pose(self, pose):
        return self._index_of_nearest_element(pose['d'], self.gripper_classes)


class ShiftIndexer(Indexer):
    def __init__(self, shift_distance):
        self.directions = ['up', 'right']
        self.shift_motions = [
            [shift_distance, 0, 0],
            [0, shift_distance, 0],
        ]

    def __len__(self) -> int:
        return len(self.directions)

    def to_action(self, action: Action) -> None:
        action.direction = self.directions[action.index]
        action.shift_motion = self.shift_motions[action.index]
        action.pose.d = 0.0
        action.type = 'shift'

    def from_action(self, action: Action, suffix='') -> int:
        return self.directions.index(action.direction)


class GraspShiftIndexer(Indexer):
    def __init__(self, gripper_classes: List[float], shift_distance):
        self.gripper_classes = gripper_classes
        self.directions = ['up', 'right']
        self.shift_motions = [
            [shift_distance, 0, 0],
            [0, shift_distance, 0],
        ]

    def __len__(self) -> int:
        return len(self.gripper_classes) + len(self.directions)

    def to_action(self, action: Action) -> None:
        pass

    def from_action(self, action: Action, suffix='') -> int:
        if action.type == 'grasp':
            return self._index_of_nearest_element(action.pose.d, self.gripper_classes)
        if action.type == 'shift':
            return len(self.gripper_classes) + self.directions.index(action.direction)
        raise Exception(f'Unknown action type {action.type}')


class LateralIndexer(Indexer):
    def __init__(self, angles, gripper_classes: List[float]):
        self.angles = angles
        self.gripper_classes = gripper_classes

    def __len__(self) -> int:
        return len(self.angles) * len(self.gripper_classes)

    def to_action(self, action: Action) -> None:
        gripper_index = action.index // len(self.angles)
        angle_index = action.index % len(self.angles)

        aff = Affine(b=self.angles[angle_index][0], c=self.angles[angle_index][1]) * action.pose
        action.pose.b = aff.b
        action.pose.c = aff.c
        action.pose.d = self.gripper_classes[gripper_index]
        action.type = 'grasp'

    def from_action(self, action: Action, suffix='') -> Optional[int]:
        local_pose = Frames.get_pose_in_image(
            action_pose=action.pose,
            image_pose=action.images[suffix]['pose'],
            reference_pose=action.images['ed-v']['pose']
        )
        angle_index = self._index_of_nearest_pose(local_pose, self.angles)
        gripper_index = self._index_of_nearest_element(action.pose.d, self.gripper_classes)
        if gripper_index is None:
            raise Exception('Gripper index is None.')
        return len(self.angles) * gripper_index + angle_index


class GraspFinalDIndexer(Indexer):
    def __init__(self, gripper_classes: List[float], final_d_classes: List[float]):
        self.gripper_classes = gripper_classes
        self.final_d_classes = final_d_classes

    def __len__(self) -> int:
        return len(self.gripper_classes) * len(self.final_d_classes)

    def to_action(self, action: Action) -> None:
        gripper_index = action.index % len(self.gripper_classes)
        action.pose.d = self.gripper_classes[gripper_index]
        action.type = 'grasp'

    def from_action(self, action: Action, suffix='') -> int:
        gripper_index = self._index_of_nearest_element(action.pose.d, self.gripper_classes)
        if 'final_pose' not in action.__dict__:
            raise Exception('Does not have a final pose.')
        final_index = 1 if action.final_pose.d > 0.035 else 0
        return len(self.gripper_classes) * final_index + gripper_index
