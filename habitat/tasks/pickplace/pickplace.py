#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import attr
import habitat_sim
import math
import numpy as np
from gym import spaces

from collections import defaultdict
from habitat.core.logging import logger
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, Simulator, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    merge_sim_episode_config,
    DistanceToGoal,
    TopDownMap,
    EpisodicGPSSensor,
    PointGoalSensor
)
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import SimulatorTaskAction, Measure
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.utils import get_habitat_sim_action, get_habitat_sim_action_str
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]

@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: List[int]


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    semantic_object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_handle: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    object_template: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    object_icon: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    motion_type: Optional[str] = attr.ib(default=None)
    is_receptacle: Optional[bool] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class GrabReleaseActionSpec:
    r"""Grab/Release action reaply data specifications that capture states
     of each grab/release action.
    """
    released_object_position: Optional[List[float]] = attr.ib(default=None)
    released_object_id: Optional[int] = attr.ib(default=None)
    released_object_handle: Optional[str] = attr.ib(default=None)
    grab_object_id: Optional[int] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectStateSpec:
    r"""Object data specifications that capture states of each object in replay state.
    """
    object_id: Optional[int] = attr.ib(default=None)
    translation: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    motion_type: Optional[str] = attr.ib(default=None)
    object_handle: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state.
    """
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)
    object_under_cross_hair: Optional[int] = attr.ib(default=None)
    action_data: Optional[GrabReleaseActionSpec] = attr.ib(default=None)
    is_grab_action: Optional[bool] = attr.ib(default=None)
    is_release_action: Optional[bool] = attr.ib(default=None)
    object_states: Optional[List[ObjectStateSpec]] = attr.ib(default=None)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)
    collision: Optional[dict] = attr.ib(default=None)
    gripped_object_id: Optional[int] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(Episode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        instruction: single natural language instruction for the task.
        reference_replay: List of keypresses which gives the reference
            actions to the goal that aligns with the instruction.
    """
    goals: List[RearrangementSpec] = attr.ib(
        default=None, validator=not_none_validator
    )
    reference_replay: List[Dict] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    objects: List[RearrangementObjectSpec] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=66,
            shape=(11,),
            dtype=np.int64,
        )

    def _get_sensor_type(self, *args:Any, **kwargs: Any):
        return SensorTypes.TOKEN_IDS

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: RearrangementEpisode,
        **kwargs
    ):
        return episode.instruction.instruction_tokens

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "demonstration"
        self.observation_space = spaces.Discrete(1)
        self.timestep = 0
        self.prev_action = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        # Fetch next action as observation
        if task.is_resetting:  # reset
            self.timestep = 1
        
        if self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
        else:
            action = 0

        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="InflectionWeightSensor")
class InflectionWeightSensor(Sensor):
    def __init__(self, config: Config, **kwargs):
        self.uuid = "inflection_weight"
        self.observation_space = spaces.Discrete(1)
        self._config = config
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        if task.is_resetting:  # reset
            self.timestep = 0
        
        inflection_weight = 1.0
        if self.timestep == 0:
            inflection_weight = 1.0
        elif self.timestep >= len(episode.reference_replay):
            inflection_weight = 1.0 
        elif episode.reference_replay[self.timestep - 1].action != episode.reference_replay[self.timestep].action:
            inflection_weight = self._config.INFLECTION_COEF
        self.timestep += 1
        return inflection_weight

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="GrippedObjectSensor")
class GrippedObjectSensor(Sensor):
    def __init__(self, *args, sim: HabitatSim, config: Config, **kwargs):
        self._sim = sim
        self.uuid = "gripped_object_id"
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(
            len(self._sim.get_existing_object_ids())
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args:Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: RearrangementEpisode,
        *args: Any,
        **kwargs
    ):
        return self._sim.gripped_object_id

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_measure
class ObjectToReceptacleDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_receptacle_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToReceptacleDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def get_navigable_position(self, position):
        if not self._sim.pathfinder.is_navigable(position):
            position = self._sim.pathfinder.snap_point(position)
            position = np.array(position).tolist()
        return position

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()
        obj_id = -1
        receptacle_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == False:
                obj_id = scene_object.object_id
            else:
                receptacle_id = scene_object.object_id
        was_nan = False
        object_position = np.array([0, 0, 0])
        if receptacle_id == -1:
            self._metric = 0.0
        elif obj_id != -1:
            object_position = np.array(
                self._sim.get_translation(obj_id)
            ).tolist()

            receptacle_position = np.array(
                self._sim.get_translation(receptacle_id)
            ).tolist()

            object_position = self.get_navigable_position(object_position)
            receptacle_position = self.get_navigable_position(receptacle_position)

            self._metric = self._geo_dist(
                object_position, receptacle_position
            )
            if self._metric == np.inf or self._metric == np.nan:
                was_nan = True
                self._metric = 2.0
        else:
            receptacle_position = np.array(
                self._sim.get_translation(receptacle_id)
            ).tolist()
            receptacle_position = self.get_navigable_position(receptacle_position)

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                agent_position, receptacle_position
            )
            if self._metric == np.inf or self._metric == np.nan:
                was_nan = True
                self._metric = 2.0
        self._metric = {
            "was_nan": was_nan,
            "metric": self._metric
        }
        # if was_nan:
        #     logger.error("object receptacle distance is nan: {} -- {} - {} -- {}".format(receptacle_id, obj_id, object_position, episode.episode_id))


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _geo_dist(self, src_pos, object_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [object_pos])

    def get_navigable_position(self, position):
        if not self._sim.pathfinder.is_navigable(position):
            position = self._sim.pathfinder.snap_point(position)
            position = np.array(position).tolist()
        return position

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()

        sim_obj_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == False:
                sim_obj_id = scene_object.object_id
        was_nan = False
        object_position = np.array([0, 0, 0])
        if sim_obj_id != -1:
            object_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()

            object_position = self.get_navigable_position(object_position)

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                agent_position, object_position
            )
            if self._metric == np.inf or self._metric == np.nan:
                was_nan = True
                self._metric = 2.0
        else:
            self._metric = 0.0
        self._metric = {
            "was_nan": was_nan,
            "metric": self._metric
        }
        if was_nan:
            logger.error("Agent object distance is inf: {} -- {} -- {}".format(sim_obj_id, object_position, episode.episode_id))


@registry.register_measure
class AgentToReceptacleDistance(Measure):
    """The measure calculates the distance of receptacle from the agent"""

    cls_uuid: str = "agent_receptacle_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToReceptacleDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _geo_dist(self, src_pos, object_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [object_pos])

    def get_navigable_position(self, position):
        if not self._sim.pathfinder.is_navigable(position):
            position = self._sim.pathfinder.snap_point(position)
            position = np.array(position).tolist()
        return position

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()

        sim_obj_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == True:
                sim_obj_id = scene_object.object_id
        was_nan = False
        if sim_obj_id != -1:
            receptacle_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()

            receptacle_position = self.get_navigable_position(receptacle_position)

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                agent_position, receptacle_position 
            )
            if self._metric == np.inf or self._metric == np.nan:
                was_nan = True
                self._metric = 2.0
        else:
            self._metric = 0.0
        self._metric = {
            "was_nan": was_nan,
            "metric": self._metric
        }


@registry.register_measure
class GoalObjectVisible(Measure):
    r"""GoalObjectVisible"""
    cls_uuid = "goal_vis_pixels"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.task_cat2mpcat40 = task_cat2mpcat40

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.update_metric(
            episode=episode, task=task, *args, **kwargs
        )

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self._metric = 0
        if "semantic" in observations:
            semantic_obs = observations["semantic"]

            if self._config.INSERTED_OBJECTS:
                object_ids = self._sim.get_existing_object_ids()
                goal_visible_pixels = 0
                semantic_object_id = self._sim.obj_id_to_semantic_obj_id_map[0]
                # If object is gripped caclulate visible pixels for receptacle
                if observations["gripped_object_id"] != -1:
                    semantic_object_id = self._sim.obj_id_to_semantic_obj_id_map[1]
                
                goal_visible_pixels += (semantic_obs == semantic_object_id).sum() # Sum over all since we're not batched
                goal_visible_area = goal_visible_pixels / semantic_obs.size
                self._metric = goal_visible_area
            else:
                # permute tensor to dimension [CHANNEL x HEIGHT X WIDTH]
                idx = self.task_cat2mpcat40[
                    observations["objectgoal"][0]
                ]  # task._dataset.category_to_task_category_id[episode.object_category], task._dataset.category_to_scene_annotation_category_id[episode.object_category], observations["objectgoal"][0]

                goal_visible_pixels = (semantic_obs == idx).sum() # Sum over all since we're not batched
                goal_visible_area = goal_visible_pixels / semantic_obs.size
                self._metric = goal_visible_area


@registry.register_measure
class Coverage(Measure):
    """Coverage
    Number of visited squares in a gridded environment
    - this is not exactly what we want to reward, but a decent starting point
    - the semantic coverage is agent-based i.e. we should reward new visuals/objects discovered
    - Note, internal grid has origin at bottom left
    EGOCENTRIC -- center + align the coverage reward.
    """
    cls_uuid: str = "coverage"

    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        self._config = config

        self._visited = None  # number of visits
        self._mini_visited = None
        self._step = None
        self._reached_count = None
        self._mini_reached = None
        self._mini_delta = 0.5
        self._grid_delta = config.GRID_DELTA
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _to_grid(self, delta, sim_x, sim_y, sim_z=0):
        # ! Note we actually get sim_x, sim_z in 2D case but that doesn't affect code
        grid_x = int((sim_x) / delta)
        grid_y = int((sim_y) / delta)
        grid_z = int((sim_z) / delta)
        return grid_x, grid_y, grid_z

    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        self._visited = {}
        self._mini_visited = {}
        self._reached_count = 0
        # Used for coverage prediction (not elegant, I know)
        self._mini_reached = 0
        self._step = 0  # Tracking episode length
        current_visit = self._visit(task, observations)
        self._metric = {
            "reached": self._reached_count,
            "mini_reached": self._mini_reached,
            "visit_count": current_visit,
            "step": self._step
        }

    def _visit(self, task, observations):
        """ Returns number of times visited current square """
        self._step += 1
        if self._config.EGOCENTRIC:
            global_loc = observations[EpisodicGPSSensor.cls_uuid]
        else:
            global_loc = self._sim.get_agent_state().position.tolist()

        mini_loc = self._to_grid(self._mini_delta, *global_loc)
        if mini_loc in self._mini_visited:
            self._mini_visited[mini_loc] += 1
        else:
            self._mini_visited[mini_loc] = 1
            self._mini_reached += 1

        grid_loc = self._to_grid(self._grid_delta, *global_loc)
        if grid_loc in self._visited:
            self._visited[grid_loc] += 1
            return self._visited[grid_loc]
        self._visited[grid_loc] = 1
        self._reached_count += 1
        return self._visited[grid_loc]

    def update_metric(
        self, episode, action, task: EmbodiedTask, observations, *args: Any, **kwargs: Any
    ):
        current_visit = self._visit(task, observations)
        self._metric = {
            "reached": self._reached_count,
            "mini_reached": self._mini_reached,
            "visit_count": current_visit,
            "step": self._step
        }


@registry.register_measure
class CoverageExplorationReward(Measure):
    # Parallels ExploreRLEnv in `environments.py`

    cls_uuid: str = "coverage_explore_reward"

    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._attenuation_penalty = 1.0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.cls_uuid,
            [
                Coverage.cls_uuid,
            ],
        )
        self._attenuation_penalty = 1.0
        self._metric = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self._attenuation_penalty *= self._config.ATTENUATION
        visit = task.measurements.measures[
            Coverage.cls_uuid
        ].get_metric()["visit_count"]
        self._metric = self._attenuation_penalty * self._config.REWARD / (visit ** self._config.VISIT_EXP)


@registry.register_measure
class ExploreThenNavReward(CoverageExplorationReward):
    # Stop count-based coverage once goal is seen

    cls_uuid: str = "explore_then_nav_reward"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_was_seen = False
        self._previous_measure = None

    def reset_metric(self, episode, task, *args, **kwargs):
        task.measurements.check_measure_dependencies(
            self.cls_uuid,
            [
                Coverage.cls_uuid,
                GoalObjectVisible.cls_uuid,
                DistanceToGoal.cls_uuid,
            ]
        )
        self._goal_was_seen = False
        self._previous_measure = None
        super().reset_metric(episode, task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._goal_was_seen:
            measure = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
            self._metric = self._previous_measure - measure
            return
        goal_vis = task.measurements.measures[GoalObjectVisible.cls_uuid].get_metric()
        if goal_vis > self._config.EXPLORE_GOAL_SEEN_THRESHOLD:
            self._goal_was_seen = True
            self._previous_measure = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        super().update_metric(*args, episode, task, **kwargs)


@registry.register_measure
class ReleaseFailed(Measure):
    r"""Grab Success - whether an object was grabbed during episode or not
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.prev_gripped_object_id = -1

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "release_failed"

    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        self._metric = 0
        self.prev_gripped_object_id = -1
        self.update_metric(episode=episode, task=task, observations=observations, action={"action": 0}, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, observations, action, *args: Any, **kwargs: Any
    ):
        action_name = task.get_action_name(action["action"])
        gripped_object_id = observations["gripped_object_id"]
        if action_name == "GRAB_RELEASE" and gripped_object_id != -1 and self.prev_gripped_object_id != -1 and gripped_object_id == self.prev_gripped_object_id:
            self._metric += 1
        self.prev_gripped_object_id = gripped_object_id


@registry.register_sensor
class AllObjectPositions(PointGoalSensor):
    cls_uuid = "all_object_positions"
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2, self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):  
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        sensor_data = np.zeros((2, self._dimensionality))
        objects = self._sim._scene_objects
        existing_object_ids = self._sim.get_existing_object_ids()

        for obj in objects:
            object_id = obj.object_id

            if object_id not in existing_object_ids:
                sensor_data[object_id] = self._compute_pointgoal(
                    agent_position, rotation_world_agent, agent_position
                )
            else:
                object_position = self._sim.get_translation(object_id)
                sensor_data[object_id] = self._compute_pointgoal(
                    agent_position, rotation_world_agent, object_position
                )
        return sensor_data


@registry.register_measure
class RearrangementReward(Measure):
    r"""RearrangementReward"""

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.step_count = 0
        self._gripped_object_count = defaultdict(int)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rearrangement_reward"

    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                AgentToObjectDistance.cls_uuid,
                AgentToReceptacleDistance.cls_uuid,
                GoalObjectVisible.cls_uuid,
                RearrangementSuccess.cls_uuid,
                Coverage.cls_uuid,
            ],
        )
        self._metric = 0
        self.step_count = 0
        self._object_was_seen = False
        self._recetpacle_was_seen = False
        self._attenuation_penalty = 1.0
        self._gripped_object_count = defaultdict(int)
        self._previous_agent_object_distance = task.measurements.measures[AgentToObjectDistance.cls_uuid].get_metric()["metric"]
        self._previous_agent_receptacle_distance = task.measurements.measures[AgentToReceptacleDistance.cls_uuid].get_metric()["metric"]
        self._previous_gripped_object_id = -1
        self._previous_exploration_area = None
        self._previous_placement_success = 0
        self._placed_success = False
        self.update_metric(episode=episode, task=task, action={"action": 0}, observations=observations, *args, **kwargs)
    
    def reward_coverage(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self._attenuation_penalty *= self._config.ATTENUATION
        visit = task.measurements.measures[
            Coverage.cls_uuid
        ].get_metric()["visit_count"]
        coverage_reward = self._attenuation_penalty * self._config.COVERAGE_REWARD / (visit ** self._config.VISIT_EXP)
        return coverage_reward

    def reward_grab_success(self, episode, task, action, observations, *args: Any, **kwargs: Any):
        reward = 0.0
        action_name = task.get_action_name(action["action"])
        current_agent_receptacle_distance = task.measurements.measures[AgentToReceptacleDistance.cls_uuid].get_metric()["metric"]
        current_object_receptacle_distance = task.measurements.measures[ObjectToReceptacleDistance.cls_uuid].get_metric()["metric"]

        if action_name != "GRAB_RELEASE":
            return reward

        if observations["gripped_object_id"] != -1 and observations["gripped_object_id"] != self._previous_gripped_object_id:
            obj_id = observations["gripped_object_id"]
            self._gripped_object_count[obj_id] += 1
            # Reward only on first grab action
            if self._gripped_object_count[obj_id] <= 1:
                reward += (
                    self._config.GRAB_SUCCESS_REWARD /
                    self._gripped_object_count[obj_id]
                )
        # Add penalty if agent drops object too far
        if self._config.ENABLE_PENALTY:
            drop_penalty_dist_threshold = (
                self._config.DROP_PENALTY_DIST_THRESHOLD
            )

            # If agent drops object too far from receptacle add penalty
            if self._previous_gripped_object_id != -1 and current_agent_receptacle_distance > drop_penalty_dist_threshold:
                reward += self._config.DROP_PENALTY
        
        # If the agent drops an object at it's successful position, 
        # give a positive reward, if it removes from a successful position,
        # give a negative reward
        if observations['gripped_object_id'] == -1:
            placement_threshold = self._config.SUCCESS_DISTANCE
            is_success = int(current_object_receptacle_distance <= placement_threshold)
            # New success will be in {-1, 0, 1}
            new_success = (is_success - self._previous_placement_success)

            reward += self._config.GRAB_SUCCESS_REWARD * new_success
            self._previous_placement_success = is_success
        
        return reward

    def reward_distance_to_object(
        self,
        episode,
        task: EmbodiedTask,
        action,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        current_agent_object_distance = task.measurements.measures[AgentToObjectDistance.cls_uuid].get_metric()["metric"]
        current_agent_receptacle_distance = task.measurements.measures[AgentToReceptacleDistance.cls_uuid].get_metric()["metric"]
        action_name = task.get_action_name(action["action"])

        if action_name != "GRAB_RELEASE" and self._previous_gripped_object_id == -1:
            agent_object_dist_reward = self._previous_agent_object_distance - current_agent_object_distance
            reward += agent_object_dist_reward

        if action_name != "GRAB_RELEASE" and self._previous_gripped_object_id != -1:
            agent_receptacle_dist_reward = self._previous_agent_receptacle_distance - current_agent_receptacle_distance
            reward += agent_receptacle_dist_reward
        return reward

    def reward_distance_to_goal(
        self,
        episode,
        task: EmbodiedTask,
        action,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        # Grab success reward
        reward += self.reward_grab_success(
            episode=episode,
            task=task,
            action=action,
            observations=observations,
            *args,
            **kwargs,
        )

        reward += self.reward_distance_to_object(
            episode=episode,
            task=task,
            action=action,
            observations=observations,
            *args,
            **kwargs,
        )
        return reward

    def reward_object_seen(
        self,
        task,
        observations
    ):
        reward = 0
        goal_object_visible = task.measurements.measures[
            GoalObjectVisible.cls_uuid
        ].get_metric()
        # Object seen threshold is small as we have a lot of small objects
        object_seen_reward_threshold = 0.001
        receptacle_seen_reward_threshold = 0.001
        if hasattr(self._config, "OBJECT_SEEN_REWARD_THRESHOLD"):
            object_seen_reward_threshold = (
                self._config.OBJECT_SEEN_REWARD_THRESHOLD
            )
        if hasattr(self._config, "RECEPTACLE_SEEN_REWARD_THRESHOLD"):
            receptacle_seen_reward_threshold = (
                self._config.RECEPTACLE_SEEN_REWARD_THRESHOLD
            )

        if (
            goal_object_visible > object_seen_reward_threshold
            and not self._object_was_seen
        ):
            self._object_was_seen = True
            reward += self._config.OBJECT_SEEN_REWARD

        if (
            goal_object_visible > receptacle_seen_reward_threshold
            and observations["gripped_object_id"] != -1
            and not self._recetpacle_was_seen
        ):
            self._recetpacle_was_seen = True
            reward += self._config.OBJECT_SEEN_REWARD
        return reward

    def reward_distance_to_object_plus_visible(
        self,
        episode,
        task: EmbodiedTask,
        action,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        # Object seen reward
        reward += self.reward_object_seen(task=task, observations=observations)

        # Grab success reward
        reward += self.reward_grab_success(
            episode=episode,
            task=task,
            action=action,
            observations=observations,
            *args,
            **kwargs,
        )

        reward += self.reward_distance_to_object(
            episode=episode,
            task=task,
            action=action,
            observations=observations,
            *args,
            **kwargs,
        )

        return reward

    def reward_explore_dt_when_object_visible(
        self,
        episode,
        task: EmbodiedTask,
        action,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        # Object seen reward
        reward += self.reward_object_seen(task=task, observations=observations)

        # Grab success reward
        reward += self.reward_grab_success(
            episode=episode,
            task=task,
            action=action,
            observations=observations,
            *args,
            **kwargs,
        )

        if not self._object_was_seen and observations["gripped_object_id"] == -1:
            reward += self.reward_coverage(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )
        else:
            reward += self.reward_distance_to_object(
                episode=episode,
                task=task,
                action=action,
                observations=observations,
                *args,
                **kwargs,
            )
        return reward

    def reward_explore_dt_when_object_or_receptacle_visible(
        self,
        episode,
        task: EmbodiedTask,
        action,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        # Object seen reward
        reward += self.reward_object_seen(task=task, observations=observations)

        # Grab success reward
        reward += self.reward_grab_success(
            episode=episode,
            task=task,
            action=action,
            observations=observations,
            *args,
            **kwargs,
        )

        # Explore until you find the object
        if not self._object_was_seen and observations["gripped_object_id"] == -1:
            # print("object not seen coverage")
            reward += self.reward_coverage(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )
        # Explore after you grab the object until you find the receptacle
        elif not self._recetpacle_was_seen and observations["gripped_object_id"] != -1:
            # print("receptacle not seen coverage")
            reward += self.reward_coverage(
                episode=episode,
                task=task,
                observations=observations,
                *args,
                **kwargs,
            )
        # Use distance reward if you saw object and receptacle when solving the task
        else:
            # print("Distance when obj/recept seen")
            reward += self.reward_distance_to_object(
                episode=episode,
                task=task,
                action=action,
                observations=observations,
                *args,
                **kwargs,
            )
        return reward

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        action,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self.step_count += 1
        # reward = self._config.SLACK_REWARD if self.step_count > 20 else 0
        reward = 0

        current_agent_object_distance = task.measurements.measures[AgentToObjectDistance.cls_uuid].get_metric()["metric"]
        current_agent_receptacle_distance = task.measurements.measures[AgentToReceptacleDistance.cls_uuid].get_metric()["metric"]

        if self._config.MODE == "DISTANCE_TO_GOAL_PLUS_VISIBLE":
            reward += self.reward_distance_to_object_plus_visible(
                episode=episode,
                task=task,
                action=action,
                observations=observations,
                *args,
                **kwargs,
            )
        elif self._config.MODE == "DISTANCE_TO_GOAL_WHEN_VISIBLE_ELSE_EXPLORE":
            reward += self.reward_explore_dt_when_object_visible(
                episode=episode,
                task=task,
                action=action,
                observations=observations,
                *args,
                **kwargs,
            )
        elif self._config.MODE == "DISTANCE_TO_OBJ_THEN_RECPT_VISIBLE_ELSE_EXPLORE":
            reward += self.reward_explore_dt_when_object_or_receptacle_visible(
                episode=episode,
                task=task,
                action=action,
                observations=observations,
                *args,
                **kwargs,
            )
        elif self._config.MODE == "DISTANCE_TO_GOAL":
            reward += self.reward_distance_to_goal(
                episode=episode,
                task=task,
                action=action,
                observations=observations,
                *args,
                **kwargs,
            )

        if task.measurements.measures[RearrangementSuccess.cls_uuid].get_metric():
            reward += self._config.SUCCESS_REWARD

        self._previous_agent_object_distance = current_agent_object_distance
        self._previous_agent_receptacle_distance = current_agent_receptacle_distance
        self._previous_gripped_object_id = observations["gripped_object_id"]
        self._metric = reward


@registry.register_measure
class RearrangementSuccess(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [ObjectToReceptacleDistance.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            ObjectToReceptacleDistance.cls_uuid
        ].get_metric()["metric"]
        object_ids = self._sim.get_existing_object_ids()

        obj_id = -1
        receptacle_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == False:
                obj_id = scene_object.object_id
            else:
                receptacle_id = scene_object.object_id

        is_object_stacked = False
        if obj_id != -1 and receptacle_id != -1:
            object_position = self._sim.get_translation(obj_id)
            receptacle_position = self._sim.get_translation(receptacle_id)

            object_y = object_position.y
            receptacle_y = receptacle_position.y + self._sim.get_object_bb_y_coord(receptacle_id)
            is_object_stacked = (object_y > receptacle_y)
        gripped_object_id = self._sim.gripped_object_id

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called # type: ignore
            and distance_to_target <= self._config.SUCCESS_DISTANCE
            and is_object_stacked
            and gripped_object_id == -1
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class RearrangementSPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, RearrangementSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            AgentToObjectDistance.cls_uuid
        ].get_metric()["metric"] + task.measurements.measures[
            ObjectToReceptacleDistance.cls_uuid
        ].get_metric()["metric"] 
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[RearrangementSuccess.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class GrabSuccess(Measure):
    r"""Grab Success - whether an object was grabbed during episode or not
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.prev_gripped_object_id = -1

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "grab_success"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        gripped_object_id = self._sim.gripped_object_id
        if gripped_object_id != -1 and gripped_object_id != self.prev_gripped_object_id:
            self._metric += 1
        self.prev_gripped_object_id = gripped_object_id


def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = episode.objects
    sim_config.freeze()

    return sim_config


@registry.register_task(name="PickPlaceTask-v0")
class PickPlaceTask(EmbodiedTask):
    r"""Language based Pick Place Task
    Goal: An agent must rearrange objects in a 3D environment
        specified by a natural language instruction.
    Usage example:
        examples/object_rearrangement_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False
    
    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return observations

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        is_object_out_of_bounds = self._sim.is_object_out_of_bounds()
        if is_object_out_of_bounds:
            logger.info("Object is OOB terminating episodes")
        return not getattr(self, "is_stop_called", False) and not is_object_out_of_bounds

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
