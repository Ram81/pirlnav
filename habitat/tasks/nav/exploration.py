# @registry.register_measure
# class ExplorationMetrics(Measure):
#     """Semantic exploration measure."""

#     cls_uuid: str = "exploration_metrics"

#     def __init__(
#         self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
#     ):
#         self._sim = sim
#         self._config = config
#         self.room_aabbs = defaultdict(list)
#         self.previous_room_stack = []
#         self.steps_between_rooms = 0
#         self.room_visitation_map = defaultdict(int)
#         self.room_revisitation_map = defaultdict(int)
#         self.room_revisitation_map_strict = defaultdict(int)
#         self.last_20_actions = []
#         self.total_left_turns = 0
#         self.total_right_turns = 0
#         self.panoramic_turns = 0
#         self.panoramic_turns_strict = 0
#         self.delta_sight_coverage = 0
#         self.prev_sight_coverage = 0
#         self.avg_delta_coverage = 0

#         super().__init__(**kwargs)

#     @staticmethod
#     def _get_uuid(*args: Any, **kwargs: Any):
#         return ExplorationMetrics.cls_uuid

#     def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
#         task.measurements.check_measure_dependencies(
#             self.uuid, [TopDownMap.cls_uuid]
#         )
#         # self.update_metric(*args, episode=episode, task=task, action={"action": 0}, **kwargs)
#         self.room_aabbs = defaultdict(list)
#         self.room_visitation_map = defaultdict(int)
#         self.room_revisitation_map = defaultdict(int)
#         self.room_revisitation_map_strict = defaultdict(int)
#         semantic_scene = self._sim.semantic_scene
#         current_room = None
#         self.steps_between_rooms = 0
#         agent_state = self._sim.get_agent_state()
#         self.last_20_actions = []
#         self.total_left_turns = 0
#         self.total_right_turns = 0
#         self.panoramic_turns = 0
#         self.panoramic_turns_strict = 0
#         self.delta_sight_coverage = 0
#         self.prev_sight_coverage = 0
#         self.avg_delta_coverage = 0
#         i = 0
#         for level in semantic_scene.levels:
#             for region in level.regions:
#                 region_name = region.category.name()
#                 if "bedroom" in region_name:
#                     region_name = region.category.name() + "_{}".format(i)
#                 aabb = region.aabb
#                 self.room_aabbs[region_name].append(aabb)

#                 if self.aabb_contains(agent_state.position, aabb):
#                     current_room = region_name
#                 i += 1

#         self._metric = self.room_revisitation_map
#         self.previous_room = current_room
    
#     def aabb_contains(self, position, aabb):
#         aabb_min = aabb.min()
#         aabb_max = aabb.max()
#         if aabb_min[0] <= position[0] and aabb_max[0] >= position[0] and aabb_min[2] <= position[2] and aabb_max[2] >= position[2] and aabb_min[1] <= position[1] and aabb_max[1] >= position[1]:
#             return True
#         return False
    
#     def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
#         return self._sim.geodesic_distance(src_pos, [goal_pos])

#     def _euclidean_distance(self, position_a, position_b):
#         return np.linalg.norm(
#             np.array(position_b) - np.array(position_a), ord=2
#         )
    
#     def _is_peeking(self, current_room):
#         prev_prev_room = None
#         prev_room = None
#         if len(self.previous_room_stack) >= 2:
#             prev_prev_room = self.previous_room_stack[-2]
#         if len(self.previous_room_stack) >= 1:
#             prev_room = self.previous_room_stack[-1]
        
#         if prev_prev_room is not None and prev_room is not None:
#             if prev_prev_room == current_room and prev_room != current_room:
#                 return True
#         return False
    

#     def get_coverage(self, info):
#         if info is None:
#             return 0
#         top_down_map = info["map"]
#         visted_points = np.where(top_down_map <= 9, 0, 1)
#         coverage = np.sum(visted_points) / self.get_navigable_area(info)
#         return coverage


#     def get_navigable_area(self, info):
#         if info is None:
#             return 0
#         top_down_map = info["map"]
#         navigable_area = np.where(((top_down_map == 1) | (top_down_map >= 10)), 1, 0)
#         return np.sum(navigable_area)


#     def get_visible_area(self, info):
#         if info is None:
#             return 0
#         fog_of_war_mask = info["fog_of_war_mask"]
#         visible_area = fog_of_war_mask.sum() / self.get_navigable_area(info)
#         if visible_area > 1.0:
#             visible_area = 1.0
#         return visible_area

#     def is_beeline(self):
#         count_move_forwards = 0
#         max_move_forwards = 0
#         for action in self.last_20_actions:
#             if action != "MOVE_FORWARD":
#                 count_move_forwards = 0
#             else:
#                 count_move_forwards += 1
#             max_move_forwards = max(max_move_forwards , count_move_forwards)
#         return (max_move_forwards / len(self.last_20_actions)) >= 0.5 

#     def update_metric(self, episode, task, action, *args: Any, **kwargs: Any):        
#         top_down_map = task.measurements.measures[
#             TopDownMap.cls_uuid
#         ].get_metric()
#         agent_state = self._sim.get_agent_state()
#         agent_position = agent_state.position
        
#         current_room = None
#         all_rooms = []
#         for region, room_aabbs in self.room_aabbs.items():
#             for aabb in room_aabbs:
#                 if self.aabb_contains(agent_position, aabb):
#                     self.room_visitation_map[region] += 1
#                     current_room = region
#                     all_rooms.append(current_room)

#         if self._is_peeking(current_room) and self.room_visitation_map[current_room] >= 1 and self.steps_between_rooms <= 10:
#             # Count total visits to the room
#             if self.room_revisitation_map[current_room] == 0:
#                 self.room_revisitation_map[current_room] += 1
#             self.room_revisitation_map[current_room] += 1
        
#         if self._is_peeking(current_room) and self.room_visitation_map[current_room] >= 1 and self.steps_between_rooms >= 8 and self.steps_between_rooms <= 14:
#             # Count total visits to the room
#             if self.room_revisitation_map_strict[current_room] == 0:
#                 self.room_revisitation_map_strict[current_room] += 1
#             self.room_revisitation_map_strict[current_room] += 1
        
#         if (len(self.previous_room_stack) == 0 or self.previous_room_stack[-1] != current_room) and current_room is not None:
#             self.previous_room_stack.append(current_room)
#             self.steps_between_rooms = 0

#         self.steps_between_rooms += 1
#         # print(top_down_map)
#         self.coverage = self.get_coverage(top_down_map)
#         self.sight_coverage = self.get_visible_area(top_down_map)

#         self.delta_sight_coverage = self.sight_coverage - self.prev_sight_coverage
#         self.prev_sight_coverage = self.sight_coverage

#         self.last_20_actions.append(task.get_action_name(action["action"]))
#         if len(self.last_20_actions) > 20:
#             self.last_20_actions.pop(0)
#         if "TURN" not in task.get_action_name(action["action"]):
#             self.total_left_turns = 0
#             self.total_right_turns = 0
#             self.delta_sight_coverage = 0
#         else:
#             if task.get_action_name(action["action"]) == "TURN_LEFT":
#                 self.total_left_turns += 1
#             elif task.get_action_name(action["action"]) == "TURN_RIGHT":
#                 self.total_right_turns += 1
#         if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.015:
#             self.panoramic_turns += 1

#         if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.01:
#             self.panoramic_turns_strict += 1
#             self.avg_delta_coverage += self.delta_sight_coverage
    
#         avg_cov = 0
#         if self.panoramic_turns_strict > 0:
#             avg_cov = self.avg_delta_coverage / self.panoramic_turns_strict

#         self._metric = {
#             "room_revisitation_map": self.room_revisitation_map,
#             "coverage": self.coverage,
#             "sight_coverage": self.sight_coverage,
#             "panoramic_turns": self.panoramic_turns,
#             "panoramic_turns_strict": self.panoramic_turns_strict,
#             "beeline": self.is_beeline(),
#             "last_20_actions": self.last_20_actions,
#             "room_revisitation_map_strict": self.room_revisitation_map_strict,
#             "delta_sight_coverage": self.delta_sight_coverage,
#             "avg_delta_coverage": avg_cov
#         }
