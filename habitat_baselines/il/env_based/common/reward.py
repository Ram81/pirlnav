

from typing import Any, Optional

import numpy as np
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal

from habitat_baselines.il.env_based.common.measures import AngleSuccess, AngleToGoal, TrainSuccess, StrictSuccess


@registry.register_measure
class SimpleReward(Measure):
    cls_uuid: str = "simple_reward"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._previous_dtg: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToGoal.cls_uuid,
                TrainSuccess.cls_uuid,
                StrictSuccess.cls_uuid,
                AngleToGoal.cls_uuid,
                AngleSuccess.cls_uuid,
            ],
        )
        self._metric = None
        self._previous_dtg = None
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        # success
        success = task.measurements.measures[TrainSuccess.cls_uuid].get_metric()
        success_reward = self._config.SUCCESS_REWARD if success else 0.0

        # distance-to-goal
        dtg = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        if self._previous_dtg is None:
            self._previous_dtg = dtg
        add_dtg = self._config.USE_DTG_REWARD
        dtg_threshold = self._config.DTG_THRESHOLD
        dtg_reward = self._previous_dtg - dtg if add_dtg and dtg < dtg_threshold else 0.0
        self._previous_dtg = dtg

        # Dense success reward
        if success and self._config.USE_STRICT_SUCCESS_REWARD and dtg < self._config.STRICT_SUCCESS_DISTANCE:
            success_reward += self._config.SUCCESS_REWARD
        
        if success and self._config.USE_STRICT_SUCCESS_REWARD_V2:
            reach_goal_within_1m = task.measurements.measures[StrictSuccess.cls_uuid].get_metric()["reached_goal_within_1m"]
            if reach_goal_within_1m:
                success_reward += self._config.SUCCESS_REWARD

        # angle success
        use_angle_success = self._config.USE_ANGLE_SUCCESS_REWARD
        angle_success = task.measurements.measures[AngleSuccess.cls_uuid].get_metric()
        angle_success_reward = (
            self._config.ANGLE_SUCCESS_REWARD if angle_success and use_angle_success else 0.0
        )

        # slack penalty
        add_slack_penalty = self._config.USE_SLACK_PENALTY
        slack_penalty = self._config.SLACK_PENALTY if add_slack_penalty else 0.0

        self._metric = (
            success_reward
            + dtg_reward
            + slack_penalty
            + angle_success_reward
        )