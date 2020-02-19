import numpy as np

from utils.param import SelectionMethod


class Epoch:
    def __init__(
            self,
            number_episodes: int,
            selection_method: SelectionMethod,
            percentage_secondary=0.0,
            secondary_selection_method=SelectionMethod.Random
    ):
        self.number_episodes = number_episodes
        self.primary_selection_method = selection_method
        self.percentage_secondary = percentage_secondary
        self.secondary_selection_method = secondary_selection_method

    def get_selection_method(self) -> SelectionMethod:
        if np.random.rand() > self.percentage_secondary:
            return self.primary_selection_method
        return self.secondary_selection_method

    @classmethod
    def get_selection_method_perform(cls, count_failed_grasps_since_last_success: int) -> SelectionMethod:
        return SelectionMethod.Max if count_failed_grasps_since_last_success == 0 else SelectionMethod.Top5

    @classmethod
    def selection_method_should_be_high(cls, method: SelectionMethod) -> bool:
        return method in [
            SelectionMethod.Max,
            SelectionMethod.Top5,
            SelectionMethod.Top5LowerBound,
            SelectionMethod.Uncertain,
            SelectionMethod.NotZero
        ]
