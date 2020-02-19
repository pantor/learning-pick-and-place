from datetime import datetime
from pathlib import Path
from typing import List

from actions.action import Action
from utils.param import Bin


class Episode:
    def __init__(self):
        self.id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
        self.save = True
        self.actions: List[Action] = []


class EpisodeHistory:
    def __init__(self):
        self.data: List[Episode] = []

    def append(self, element: Episode):
        self.data.append(element)

    def total(self) -> int:
        return len(self.data)

    def iterate_actions(self, filter_cond=None, stop_cond=None):
        for e in self.data[::-1]:
            if not e.actions:
                continue
            if stop_cond and stop_cond(e.actions[0]):
                break
            if filter_cond and not filter_cond(e.actions[0]):
                continue
            yield e.actions[0]

    def total_reward(self, action_type=None):
        return sum(a.reward for a in self.iterate_actions(
            filter_cond=lambda a: a.type == action_type if action_type else True
        ))

    def failed_grasps_since_last_success_in_bin(self, current_bin: Bin):
        return sum(1 for _ in self.iterate_actions(
            filter_cond=lambda a: a.type == 'grasp' and a.reward == 0.0,
            stop_cond=lambda a: a.bin is not current_bin or a.reward == 1.0,
        ))

    def successful_grasps_in_bin(self, current_bin: Bin):
        return sum(1 for _ in self.iterate_actions(
            filter_cond=lambda a: a.reward == 1.0,
            stop_cond=lambda a: a.bin is not current_bin,
        ))

    def save_grasp_rate_evaluation(self, file_path: Path) -> None:
        last_bin = self.data[-1].actions[0].bin

        number_episodes = sum(1 for _ in self.iterate_actions(stop_cond=lambda a: a.bin is not last_bin))
        mean_episode_time = sum(action.execution_time for action in self.iterate_actions(
            stop_cond=lambda a: a.bin is not last_bin
        )) / number_episodes
        shifts_since_last_success = sum(1 for _ in self.iterate_actions(
            filter_cond=lambda a: a.type == 'shift',
            stop_cond=lambda a: a.bin is not last_bin,
        ))

        with open(file_path, 'a+') as f:
            f.write(f'{number_episodes}, {mean_episode_time:0.3f}, {shifts_since_last_success}\n')

    def save_grasp_rate_prediction_step_evaluation(self, file_path: Path) -> None:
        with open(file_path, 'a+') as f:
            for a in self.iterate_actions():
                f.write(f'{a.step}, {a.estimated_reward:0.3f}, {a.estimated_reward_std:0.3f}, {a.reward:0.3f}, {a.bin}\n')
                break
