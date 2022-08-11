from typing import Any

import numpy as np
import quaternion

from habitat import logger
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, Success, DistanceToGoal
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)


@registry.register_measure
class AngleToGoal(Measure):
    """The measure calculates an angle towards the goal. Note: this measure is
    only valid for single goal tasks (e.g., ImageNav)
    """

    cls_uuid: str = "angle_to_goal"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any):
        self._metric = None
        dependencies = [StrictSuccess.cls_uuid]
        task.measurements.check_measure_dependencies(self.uuid, dependencies)
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode: NavigationEpisode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        current_rotation = self._sim.get_agent_state().rotation
        if not isinstance(current_rotation, quaternion.quaternion):
            current_rotation = quaternion_from_coeff(current_rotation)

        goal_rotation = task.measurements.measures[StrictSuccess.cls_uuid].get_metric()["goal_rotation"]

        if goal_rotation is not None:            
            if not isinstance(goal_rotation, quaternion.quaternion):
                goal_rotation = quaternion_from_coeff(goal_rotation)

            self._metric = angle_between_quaternions(current_rotation, goal_rotation)
        else:
            self._metric = 0


@registry.register_measure
class AngleSuccess(Measure):
    """Weather or not the agent is within an angle tolerance."""

    cls_uuid: str = "angle_success"

    def __init__(self, config: Config, *args: Any, **kwargs: Any):
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        dependencies = [AngleToGoal.cls_uuid]
        if self._config.USE_TRAIN_SUCCESS:
            dependencies.append(TrainSuccess.cls_uuid)
        else:
            dependencies.append(Success.cls_uuid)
        task.measurements.check_measure_dependencies(self.uuid, dependencies)
        self.update_metric(task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        if self._config.USE_TRAIN_SUCCESS:
            success = task.measurements.measures[TrainSuccess.cls_uuid].get_metric()
        else:
            success = task.measurements.measures[Success.cls_uuid].get_metric()
        angle_to_goal = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()

        if success and np.rad2deg(angle_to_goal) < self._config.SUCCESS_ANGLE:
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class TrainSuccess(Success):
    r"""Whether or not the agent succeeded at its task
    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "train_success"

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class StrictSuccess(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "strict_success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        self._closest_point = None
        self._episode_view_points = [
            view_point.agent_state.position
            for goal in episode.goals
            for view_point in goal.view_points
        ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            closest_points = None
            distance_to_target, closest_points = self._sim.geodesic_distance(
                current_position, self._episode_view_points, episode, return_points=True
            )

            # Using closest points to current goal find the goal rotation
            reached_goal_within_1m = False
            if distance_to_target < self._config.SUCCESS_DISTANCE:
                for goal in episode.goals:
                    for view_point in goal.view_points:
                        all_close = [self._euclidean_distance(view_point.agent_state.position, closest_point) for closest_point in closest_points]
                        if view_point.within_1m and min(all_close) < self._config.VIEW_POINT_THRESHOLD:
                            reached_goal_within_1m = True
                            break

            self._previous_position = current_position
            self._metric = {
                "dtg": distance_to_target,
                "points": closest_points,
                "reached_goal_within_1m": reached_goal_within_1m,
            }
