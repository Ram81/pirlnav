#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
from typing import List, Optional

import numpy as np
import pydash

import habitat_sim
from habitat.core.dataset import SceneState
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.datasets.pointnav.pointnav_generator import (
    ISLAND_RADIUS_LIMIT,
    _ratio_sample_rate,
)
from habitat.datasets.utils import get_action_shortest_path
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from habitat.tasks.utils import compute_pixel_coverage
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)

# from habitat.utils.geometry_utils import direction_to_quaternion
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_to_angle_axis,
    quat_to_coeffs,
)


def _direction_to_quaternion(direction_vector: np.array):
    origin_vector = np.array([0, 0, -1])
    return quaternion_from_two_vectors(origin_vector, direction_vector)


def _get_multipath(sim: HabitatSim, start, ends):
    multi_goal = habitat_sim.MultiGoalShortestPath()
    multi_goal.requested_start = start
    multi_goal.requested_ends = ends
    sim.pathfinder.find_path(multi_goal)
    return multi_goal


def _get_action_shortest_path(
    sim: HabitatSim, start_pos, start_rot, goal_pos, goal_radius=0.05
):
    sim.set_agent_state(start_pos, start_rot, reset_sensors=True)
    greedy_follower = sim.make_greedy_follower()
    return greedy_follower.find_path(goal_pos)


def is_compatible_episode(
    s,
    t,
    sim: HabitatSim,
    goals: List[ObjectGoal],
    near_dist,
    far_dist,
    geodesic_to_euclid_ratio,
):
    FAIL_TUPLE = False, 0, 0, [], [], [], []
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return FAIL_TUPLE
    s = np.array(s)
    # distance = np.array(list(map(lambda pos: sim.geodesic_distance(s, pos), t)))

    # assert(len(goals_repeated) == len(t))
    # TWO TARGETS MAY BE BETWEEN TWO GOALS
    goal_targets = (
        [vp.agent_state.position for vp in goal.view_points] for goal in goals
    )

    # old_school
    # closest_goal_targets = (min((sim.geodesic_distance(s, vp), vp) for vp in vps) for vps in goal_targets)
    # closest_goal_targets, goals_sorted = zip(*sorted(zip(closest_goal_targets, goals)))
    # d_separation, closest_target = closest_goal_targets[0]
    # closest_goal_targets = [_get_multipath(sim ,s, goal_targets)]
    closest_goal_targets = (
        sim.geodesic_distance(s, vps) for vps in goal_targets
    )
    closest_goal_targets, goals_sorted = zip(
        *sorted(zip(closest_goal_targets, goals), key=lambda x: x[0])
    )
    d_separation = closest_goal_targets[0]

    if (
        np.inf in closest_goal_targets
        or not near_dist <= d_separation <= far_dist
    ):
        return FAIL_TUPLE

    # shortest_path = sim.get_straight_shortest_path_points(s, closest_target)
    shortest_path = None
    # shortest_path = closest_goal_targets[0].points
    shortest_path_pos = shortest_path
    euclid_dist = np.linalg.norm(s - goals_sorted[0].position)
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return FAIL_TUPLE

    # geodesic_distances, _ = zip(*closest_goal_targets)
    geodesic_distances = closest_goal_targets

    # euclid_distances_target = np.linalg.norm(np.array([goal.position for goal in goals]) - shortest_path[-1:], axis=1)
    # _, goals_sorted = zip(*sorted(zip(euclid_distances_target, goals)))
    # goal_index = np.argmin(euclid_distances_target)
    # goals_sorted = [goals[goal_index]] + goals[:goal_index] + goals[goal_index+1:]

    angle = np.random.uniform(0, 2 * np.pi)
    source_rotation = [
        0,
        np.sin(angle / 2),
        0,
        np.cos(angle / 2),
    ]  # Pick random starting rotation

    # try:
    #     action_shortest_path = _get_action_shortest_path(
    #         sim, s, source_rotation, shortest_path[-1]
    #     )
    #     if action_shortest_path == None:
    #         return FAIL_TUPLE
    #     shortest_path = (
    #         action_shortest_path
    #     )
    #     # [ShortestPathPoint(point, [0,0,0], action) for point, action in zip(shortest_path, action_shortest_path)]
    # except GreedyFollowerError:
    #     print(
    #         "Could not find path between %s and %s"
    #         % (str(s), str(shortest_path[-1]))
    #     )
    #     return FAIL_TUPLE
    ending_state = None
    # ending_state = closest_goal_targets[0].points[-1]  # sim.get_agent_state()
    # ending_position = ending_state.position
    #
    # e_q = ending_state.rotation  # We presume the agent is upright
    # # print(shortest_path)
    # goal_direction = goals_sorted[0].position - ending_position
    # goal_direction[1] = 0
    # a_q = _direction_to_quaternion(goal_direction)
    # quat_delta = e_q - a_q
    # # The angle between the two quaternions should be how much to turn
    # theta, _ = quat_to_angle_axis(quat_delta)
    # if theta < 0 or theta > np.pi:
    #     turn = HabitatSimActions.TURN_LEFT
    #     if theta > np.pi:
    #         theta = 2 * np.pi - theta
    # else:
    #     turn = HabitatSimActions.TURN_RIGHT
    # turn_angle = np.deg2rad(sim.config.TURN_ANGLE)
    # num_of_turns = int(theta / turn_angle + 0.5)
    # shortest_path += [turn] * num_of_turns
    # angle = angle_between_quaternions(a_q, e_q)

    # if len(shortest_path) > 750:
    #     print("ERROR SHORTEST PATH IS TOO LONG")
    #     return FAIL_TUPLE

    # shortest_path = get_action_shortest_path(sim,
    #     s,
    #     source_rotation,
    #     goals_sorted[0].position,
    #     max_episode_steps=750)
    # #CANNOT CATCH ERRORS
    # if shortest_path == None:
    #     return FAIL_TUPLE
    # Make it so it doesn't have to see the object initially
    # TODO Consider turning this check back on later?
    # num_rotation_attempts = 20
    # for attempt in range(num_rotation_attempts):
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

    #     obs = sim.get_observations_at(s, source_rotation)

    #     if all(
    #         (compute_pixel_coverage(obs["semantic"], g.object_id) / g.best_iou) < 0.01
    #         for g in goals_sorted
    #     ):
    #         break

    # if attempt == num_rotation_attempts:
    #     return FAIL_TUPLE

    return (
        True,
        d_separation,
        euclid_dist,
        source_rotation,
        geodesic_distances,
        goals_sorted,
        shortest_path,
        ending_state,
    )


def build_goal(
    sim: HabitatSim,
    object_id: int,
    object_name_id: int,
    object_category_name: str,
    object_category_id: int,
    object_position,
    object_aabb: habitat_sim.geo.BBox,
    object_obb: habitat_sim.geo.OBB,
    object_gmobb: habitat_sim.geo.OBB,
    cell_size: float = 1.0,
    grid_radius: float = 10.0,
    turn_radians: float = np.pi / 9,
    max_distance: float = 1.0,
):
    object_position = object_aabb.center
    eps = 1e-5

    x_len, y_len, z_len = object_aabb.sizes / 2.0 + max_distance
    x_bxp = np.arange(-x_len, x_len + eps, step=cell_size) + object_position[0]
    # y_bxp = np.arange(-y_len, y_len + eps, step=1) + object_position[1]
    # y = np.arange(-y_len, y_len, step=cell_size)
    z_bxp = np.arange(-z_len, z_len + eps, step=cell_size) + object_position[2]
    # print('Original obb %s' % object_obb.center )
    # #print('Original calculated: %f %f %f' % (x_len, y_len, z_len))
    # print('Obb calculated: %s ' % obb.center)
    # print('Obb calculated: %s' % obb.half_extents)
    # print('Obb calculated sizes: %s' % (obb.sizes / 2.0))
    # print('AABB: %s %s' % (object_aabb.center, object_aabb.sizes))
    candiatate_poses = [
        np.array([x, object_position[1], z])
        for x, z in itertools.product(x_bxp, z_bxp)
    ]

    # x_len, y_len, z_len = obb.half_extents + MAX_DISTANCE

    # x_arr = np.arange(-x_len, x_len + eps, step=cell_size, dtype=np.float32) / obb.half_extents[0]
    # z_arr = np.arange(-z_len, z_len + eps, step=cell_size, dtype=np.float32) / obb.half_extents[2]

    # candiatate_poses = [(obb.local_to_world @ np.array([x, 0, z, 1]))[:-1] for x, z in itertools.product(x_arr, z_arr)]
    # print('New Bounding Area: %s' % np.array([(obb.local_to_world @ np.array([x, y, z, 1]))[:-1] for x,y,z in
    #     itertools.product(*([[-1, 1]] * 3))]))
    # #print('Candiate Poses %s' % str(candiatate_poses))
    # print('Candiate Poses variance: %s ' % str(habitat_sim.geo.compute_gravity_aligned_MOBB(habitat_sim.geo.GRAVITY, candiatate_poses).sizes))
    # #candiatate_poses = [np.array([p[0], obb.center[1], p[2]]) for p in candiatate_poses] #if obb.distance(np.array([p[0], obb.center[1], p[2]])) <= MAX_DISTANCE]
    # print(candiatate_poses)
    # obb = habitat_sim.geo.OBB(object_aabb)
    # thetas = np.arange(0, 2 * np.pi, step=turn_radians)
    def down_is_navigable(pt):
        pf = sim.pathfinder

        delta_y = 0.05
        max_steps = int(2 / delta_y)
        step = 0
        is_navigable = pf.is_navigable(pt, 2)
        while not is_navigable:
            pt[1] -= delta_y
            is_navigable = pf.is_navigable(pt)
            step += 1
            if step == max_steps:
                return False
        return True

    def _get_iou(x, y, z):
        pt = np.array([x, y, z])

        if not (
            object_gmobb.distance(pt) <= max_distance
            and habitat_sim.geo.OBB(object_aabb).distance(pt) <= max_distance
        ):
            return -0.5, pt, None

        if not down_is_navigable(pt):
            return -1.0, pt, None
        # if not pf.is_navigable(pt, 2) and not pf.is_navigable(np.array(pf.snap_point(pt)), 2):
        #     # other_points = [(pf.is_navigable(p), p) for p in itertools.product([-x,x], [-y,y], [-z,z])]
        #     # if any(t[0] for t in other_points):
        #     #     print(pt)
        #     #     print(other_points)
        #     return -1.0, pt, None
        pf = sim.pathfinder
        pt = np.array(pf.snap_point(pt))

        goal_direction = object_position - pt
        # print(goal_direction)
        goal_direction[1] = 0

        q = _direction_to_quaternion(goal_direction)
        # POSITION MAY BE UNREACHABLE
        # goal_angle, _ = quat_to_angle_axis(goal_rot)
        # theta = goal

        # print(pt)
        # print(pt)
        # print('*' * 100)

        # q = quat_from_angle_axis(theta, habitat_sim.geo.UP)
        cov = 0
        agent = sim.get_agent(0)
        for act in [
            HabitatSimActions.LOOK_DOWN,
            HabitatSimActions.LOOK_UP,
            HabitatSimActions.LOOK_UP,
        ]:
            # if act is not None:
            agent.act(act)
            for v in agent._sensors.values():
                v.set_transformation_from_spec()
            obs = sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
            cov += compute_pixel_coverage(obs["semantic"], object_id)
        from habitat.utils.visualizations.utils import observations_to_image
        import imageio
        import os

        obs = sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
        imageio.imsave(
            os.path.join(
                "data/images/objnav_dataset_gen",
                f"{object_name_id}_{object_id}_{x}_{z}_.png",
            ),
            observations_to_image(obs, info={}),
        )

        return cov, pt, q

    # candiatate_poses = [
    #     object_position + np.array([x, 0, z])
    #     #(obb.local_to_world @ np.array([x, y_len, z, 1]))[:-1]
    #     for x, z in itertools.product(x, z)
    # ]
    # PROPER
    # END
    # candiatate_poses = (object_obb.local_to_world @ np.array([x, 0, z, 1]) for x, y, z in itertools.product(x, z))
    # candiatate_poses = [p[:3] for p in candiatate_poses]
    # candiatate_poses = [(x, y, z) for x, z in itertools.product(x, z)]
    candiatate_poses_ious = list(_get_iou(*pos) for pos in candiatate_poses)

    # print(np.mean([a[1] - object_position for a in candiatate_poses_ious if np.any(a[1] != None) and a[0] != 0], axis=0))
    best_iou = (
        max(v[0] for v in candiatate_poses_ious)
        if len(candiatate_poses_ious) != 0
        else 0
    )
    if best_iou <= 0.0:
        print(
            f"No view points found for {object_name_id}_{object_id}: {best_iou}"
        )
        return None

    keep_thresh = 0  # 1/(256**2)#0.75 * best_iou

    view_locations = [
        ObjectViewLocation(
            AgentState(pt.tolist(), quat_to_coeffs(q).tolist()), iou
        )
        for iou, pt, q in candiatate_poses_ious
        if iou is not None and iou > keep_thresh
    ]
    import habitat.datasets.object_nav.create.debug_utils as debug_utils

    debug_utils.plot_area(
        candiatate_poses_ious,
        [v.agent_state.position for v in view_locations],
        [object_position],
        object_category_name + object_name_id,
    )

    view_locations = sorted(view_locations, reverse=True, key=lambda v: v.iou)
    if len(view_locations) == 0:
        print(
            f"No valid views found for {object_name_id}_{object_id}: {best_iou}"
        )
        return None
    # for view in view_locations:
    #     obs = sim.get_observations_at(
    #         view.agent_state.position, view.agent_state.rotation
    #     )

    #     from habitat.utils.visualizations.utils import (
    #        observations_to_image,
    #        get_image_with_obj_overlay,
    #     )
    #     import imageio
    #     import os

    #     obs["rgb"] = get_image_with_obj_overlay(obs, [object_id])
    #     imageio.imsave(
    #        os.path.join(
    #            "data/images/objnav_dataset_gen/",
    #            f"{object_name_id}_{object_id}_{view.iou}_{view.agent_state.position}.png",
    #        ),
    #        observations_to_image(obs, info={}).astype(np.uint8),
    #     )
    goal = ObjectGoal(
        position=object_position.tolist(),
        view_points=view_locations,
        object_id=object_id,
        object_category=object_category_name,
        object_name=object_name_id,
    )

    return goal


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    goals,
    shortest_paths=None,
    scene_state=None,
    info=None,
):
    return ObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        object_category=goals[0].object_category,
        # scene_state=scene_state,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


def generate_objectnav_episode(
    sim: HabitatSim,
    goals: List[ObjectGoal],
    scene_state: SceneState = None,
    num_episodes: int = -1,
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_target: int = 1000,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        # target_positions = (
        #     pydash.chain()
        #     .map(lambda g: g.view_points)
        #     .flatten().map(lambda v: v.agent_state.position)(goals)
        # )
        # Cache this transformation
        target_positions = np.array(
            list(
                itertools.chain(
                    *(
                        (
                            view_point.agent_state.position
                            for view_point in g.view_points
                        )
                        for g in goals
                    )
                )
            )
        )

        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            if (
                source_position is None
                or np.any(np.isnan(source_position))
                or not sim.is_navigable(source_position)
            ):
                raise RuntimeError("Unable to find valid starting location")
            if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
                continue
            compat_outputs = is_compatible_episode(
                source_position,
                target_positions,
                sim,
                goals,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            is_compatible = compat_outputs[0]

            if is_compatible:
                is_compatible, dist, euclid_dist, source_rotation, geodesic_distances, goals_sorted, shortest_path, ending_state = (
                    compat_outputs
                )
                shortest_paths = None #[shortest_path]

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=sim.habitat_config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    shortest_paths=shortest_paths,
                    scene_state=scene_state,
                    info={
                        "geodesic_distance": dist,
                        "euclidean_distance": euclid_dist,
                        "closest_goal_object_id": goals_sorted[0].object_id,
                        # "navigation_bounds": sim.pathfinder.get_bounds(),
                        # "best_viewpoint_position": ending_state.position,
                    },
                    goals=goals_sorted,
                )

                episode_count += 1
                yield episode
                break

        if retry == number_retries_per_target:
            raise RuntimeError("Unable to find valid starting location")


def generate_objectnav_episode_with_added_objects(
    sim: HabitatSim,
    objects: List,
    goal_category: str,
    goals: List[ObjectGoal],
    scene_state: SceneState = None,
    num_episodes: int = -1,
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_target: int = 1000,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        # target_positions = (
        #     pydash.chain()
        #     .map(lambda g: g.view_points)
        #     .flatten().map(lambda v: v.agent_state.position)(goals)
        # )
        # Cache this transformation
        target_positions = np.array(
            list(
                itertools.chain(
                    *(
                        (
                            view_point.agent_state.position
                            for view_point in g.view_points
                        )
                        for g in goals
                    )
                )
            )
        )

        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            if (
                source_position is None
                or np.any(np.isnan(source_position))
                or not sim.is_navigable(source_position)
            ):
                raise RuntimeError("Unable to find valid starting location")
            if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
                continue
            compat_outputs = is_compatible_episode(
                source_position,
                target_positions,
                sim,
                goals,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            is_compatible = compat_outputs[0]

            if is_compatible:
                is_compatible, dist, euclid_dist, source_rotation, geodesic_distances, goals_sorted, shortest_path, ending_state = (
                    compat_outputs
                )
                shortest_paths = [shortest_path]

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=sim.habitat_config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    shortest_paths=shortest_paths,
                    scene_state=scene_state,
                    info={
                        "geodesic_distance": dist,
                        "euclidean_distance": euclid_dist,
                        "closest_goal_object_id": goals_sorted[0].object_id,
                        # "navigation_bounds": sim.pathfinder.get_bounds(),
                        # "best_viewpoint_position": ending_state.position,
                    },
                    goals=goals_sorted,
                )

                episode_count += 1
                yield episode
                break

        if retry == number_retries_per_target:
            raise RuntimeError("Unable to find valid starting location")
