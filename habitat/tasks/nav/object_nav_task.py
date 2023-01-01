# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import clip
from typing import Any, List, Optional

import attr
from cv2 import log
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import SceneState
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    DistanceToGoal,
    EpisodicGPSSensor,
    TopDownMap
)

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass
from collections import defaultdict


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

task_cat2hm3dcat40 = [
    3,  # ('chair', 2, 0)
    11,  # ('bed', 10, 6)
    14,  # ('plant', 13, 8)
    18,  # ('toilet', 17, 10),
    22,  # ('tv_monitor', 21, 13)
    10,  # ('sofa', 9, 5),
]

task_cat2hm3d_shapeconv_cat = [
    1,  # ('chair', 2, 0)
    7,  # ('bed', 10, 6)
    9,  # ('plant', 13, 8)
    11,  # ('toilet', 17, 10),
    14,  # ('tv_monitor', 21, 13)
    6,  # ('sofa', 9, 5),
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}


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
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state: Optional[List[SceneState]] = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "mp3d"
    scene_dataset_config: Optional[str] = ""
    additional_obj_config_paths: Optional[List] = []
    attempts: Optional[int] = 1

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]
    within_1m: Optional[bool] = False


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.GOAL_SPEC_MAX_VAL - 1
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )
            logger.info("max object cat: {}".format(max_value))
            logger.info("cats: {}".format(self._dataset.category_to_task_category_id.values()))

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    _is_episode_active: bool
    _prev_action: int

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False

    def overwrite_sim_config(self, sim_config, episode):
        super().overwrite_sim_config(sim_config, episode)

        sim_config.defrost()
        sim_config.scene_state = episode.scene_state
        sim_config.freeze()
        
        return sim_config

    def _check_episode_is_active(self,  action, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@registry.register_measure
class MinDistanceToGoal(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "min_distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._min_distance_to_goal = 100

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._min_distance_to_goal = 100
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        self._metric = min(self._min_distance_to_goal, distance_to_target)
        self._min_distance_to_goal = self._metric



@registry.register_sensor
class ObjectGoalPromptSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal_prompt"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        self._clip_goal_cache = np.load(config.CACHE_PATH)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    # def _get_observation_space(self, *args: Any, **kwargs: Any):
    #     sensor_shape = self._clip_goal_cache[0].shape
    #     max_value = np.max(self._clip_goal_cache)
    #     min_value = np.max(self._clip_goal_cache)
    #     if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
    #         max_value = max(
    #             self._dataset.category_to_task_category_id.values()
    #         )
    #         logger.info("max object cat: {}".format(max_value))
    #         logger.info("cats: {}".format(self._dataset.category_to_task_category_id.values()))

    #     return spaces.Box(
    #         low=min_value, high=max_value, shape=sensor_shape, dtype=self._clip_goal_cache.dtype
    #     )

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=np.inf, shape=(1, 77,), dtype=np.int64)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None

        category_name = episode.object_category
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            # object_category_id = self._dataset.category_to_task_category_id[category_name]
            # return self._clip_goal_cache[object_category_id]
            return clip.tokenize(category_name).numpy()
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


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
class ExplorationMetrics(Measure):
    """Semantic exploration measure."""

    cls_uuid: str = "exploration_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.coverage = 0
        self.sight_coverage = 0

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ExplorationMetrics.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [TopDownMap.cls_uuid]
        )

        self._metric = {
            "coverage": 0,
            "sight_coverage": 0,
        }

    def get_coverage(self, info):
        if info is None:
            return 0
        top_down_map = info["map"]
        visted_points = np.where(top_down_map <= 9, 0, 1)
        coverage = np.sum(visted_points) / self.get_navigable_area(info)
        return coverage

    def get_navigable_area(self, info):
        if info is None:
            return 0
        top_down_map = info["map"]
        navigable_area = np.where(((top_down_map == 1) | (top_down_map >= 10)), 1, 0)
        return np.sum(navigable_area)

    def get_visible_area(self, info):
        if info is None:
            return 0
        fog_of_war_mask = info["fog_of_war_mask"]
        visible_area = fog_of_war_mask.sum() / self.get_navigable_area(info)
        if visible_area > 1.0:
            visible_area = 1.0
        return visible_area

    def update_metric(self, episode, task, action, *args: Any, **kwargs: Any):        
        top_down_map = task.measurements.measures[
            TopDownMap.cls_uuid
        ].get_metric()

        self.coverage = self.get_coverage(top_down_map)
        self.sight_coverage = self.get_visible_area(top_down_map)

        self._metric = {
            "coverage": self.coverage,
            "sight_coverage": self.sight_coverage,
        }


@registry.register_measure
class BehaviorMetrics(Measure):
    """Human-like behaviors measure."""

    cls_uuid: str = "behavior_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.room_aabbs = defaultdict(list)
        self.previous_room_stack = []
        self.steps_between_rooms = 0
        self.room_visitation_map = defaultdict(int)
        self.room_revisitation_map = defaultdict(int)
        self.room_revisitation_map_strict = defaultdict(int)
        self.last_20_actions = []
        self.total_left_turns = 0
        self.total_right_turns = 0
        self.panoramic_turns = 0
        self.panoramic_turns_strict = 0
        self.delta_sight_coverage = 0
        self.prev_sight_coverage = 0
        self.avg_delta_coverage = 0

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return BehaviorMetrics.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [TopDownMap.cls_uuid]
        )

        self.room_aabbs = defaultdict(list)
        self.room_visitation_map = defaultdict(int)
        self.room_revisitation_map = defaultdict(int)
        self.room_revisitation_map_strict = defaultdict(int)
        semantic_scene = self._sim.semantic_scene
        current_room = None
        self.steps_between_rooms = 0
        agent_state = self._sim.get_agent_state()
        self.last_20_actions = []
        self.total_left_turns = 0
        self.total_right_turns = 0
        self.panoramic_turns = 0
        self.panoramic_turns_strict = 0
        self.delta_sight_coverage = 0
        self.prev_sight_coverage = 0
        self.avg_delta_coverage = 0
        i = 0
        for level in semantic_scene.levels:
            for region in level.regions:
                region_name = region.category.name()
                if "bedroom" in region_name:
                    region_name = region.category.name() + "_{}".format(i)
                aabb = region.aabb
                self.room_aabbs[region_name].append(aabb)

                if self.aabb_contains(agent_state.position, aabb):
                    current_room = region_name
                i += 1

        self._metric = {}
        self.previous_room = current_room
    
    def aabb_contains(self, position, aabb):
        aabb_min = aabb.min()
        aabb_max = aabb.max()
        if aabb_min[0] <= position[0] and aabb_max[0] >= position[0] and aabb_min[2] <= position[2] and aabb_max[2] >= position[2] and aabb_min[1] <= position[1] and aabb_max[1] >= position[1]:
            return True
        return False
    
    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def get_coverage(self, info):
        if info is None:
            return 0
        top_down_map = info["map"]
        visted_points = np.where(top_down_map <= 9, 0, 1)
        coverage = np.sum(visted_points) / self.get_navigable_area(info)
        return coverage

    def get_navigable_area(self, info):
        if info is None:
            return 0
        top_down_map = info["map"]
        navigable_area = np.where(((top_down_map == 1) | (top_down_map >= 10)), 1, 0)
        return np.sum(navigable_area)

    def get_visible_area(self, info):
        if info is None:
            return 0
        fog_of_war_mask = info["fog_of_war_mask"]
        visible_area = fog_of_war_mask.sum() / self.get_navigable_area(info)
        if visible_area > 1.0:
            visible_area = 1.0
        return visible_area
    
    def _is_peeking(self, current_room):
        prev_prev_room = None
        prev_room = None
        if len(self.previous_room_stack) >= 2:
            prev_prev_room = self.previous_room_stack[-2]
        if len(self.previous_room_stack) >= 1:
            prev_room = self.previous_room_stack[-1]
        
        if prev_prev_room is not None and prev_room is not None:
            if prev_prev_room == current_room and prev_room != current_room:
                return True
        return False

    def is_beeline(self):
        count_move_forwards = 0
        max_move_forwards = 0
        for action in self.last_20_actions:
            if action != "MOVE_FORWARD":
                count_move_forwards = 0
            else:
                count_move_forwards += 1
            max_move_forwards = max(max_move_forwards, count_move_forwards)
        return (max_move_forwards / len(self.last_20_actions)) >= 0.5 

    def update_metric(self, episode, task, action, *args: Any, **kwargs: Any):        
        top_down_map = task.measurements.measures[
            TopDownMap.cls_uuid
        ].get_metric()
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        
        current_room = None
        all_rooms = []
        for region, room_aabbs in self.room_aabbs.items():
            for aabb in room_aabbs:
                if self.aabb_contains(agent_position, aabb):
                    self.room_visitation_map[region] += 1
                    current_room = region
                    all_rooms.append(current_room)

        if self._is_peeking(current_room) and self.room_visitation_map[current_room] >= 1 and self.steps_between_rooms <= 10:
            # Count total visits to the room
            if self.room_revisitation_map[current_room] == 0:
                self.room_revisitation_map[current_room] += 1
            self.room_revisitation_map[current_room] += 1
        
        if self._is_peeking(current_room) and self.room_visitation_map[current_room] >= 1 and self.steps_between_rooms >= 8 and self.steps_between_rooms <= 14:
            # Count total visits to the room
            if self.room_revisitation_map_strict[current_room] == 0:
                self.room_revisitation_map_strict[current_room] += 1
            self.room_revisitation_map_strict[current_room] += 1
        
        if (len(self.previous_room_stack) == 0 or self.previous_room_stack[-1] != current_room) and current_room is not None:
            self.previous_room_stack.append(current_room)
            self.steps_between_rooms = 0

        self.steps_between_rooms += 1
        self.coverage = self.get_coverage(top_down_map)
        self.sight_coverage = self.get_visible_area(top_down_map)

        self.delta_sight_coverage = self.sight_coverage - self.prev_sight_coverage
        self.prev_sight_coverage = self.sight_coverage

        self.last_20_actions.append(task.get_action_name(action["action"]))
        if len(self.last_20_actions) > 20:
            self.last_20_actions.pop(0)
        if "TURN" not in task.get_action_name(action["action"]):
            self.total_left_turns = 0
            self.total_right_turns = 0
            self.delta_sight_coverage = 0
        else:
            if task.get_action_name(action["action"]) == "TURN_LEFT":
                self.total_left_turns += 1
            elif task.get_action_name(action["action"]) == "TURN_RIGHT":
                self.total_right_turns += 1
        if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.015:
            self.panoramic_turns += 1

        if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.01:
            self.panoramic_turns_strict += 1
            self.avg_delta_coverage += self.delta_sight_coverage

        if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.015:
            self.panoramic_turns += 1
    
        avg_cov = 0
        if self.panoramic_turns_strict > 0:
            avg_cov = self.avg_delta_coverage / self.panoramic_turns_strict

        self._metric = {
            "room_revisitation_map": self.room_revisitation_map,
            "coverage": self.coverage,
            "sight_coverage": self.sight_coverage,
            "panoramic_turns": self.panoramic_turns,
            "panoramic_turns_strict": self.panoramic_turns_strict,
            "beeline": self.is_beeline(),
            "last_20_actions": self.last_20_actions,
            "room_revisitation_map_strict": self.room_revisitation_map_strict,
            "delta_sight_coverage": self.delta_sight_coverage,
            "avg_delta_coverage": avg_cov
        }
