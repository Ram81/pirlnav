import random
from time import time
from typing import List

import numpy as np
from pydash import py_

import habitat
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal
from habitat.tasks.nav.shortest_path_follower import (  # mp3d_direction_to_hsim_quaternion,
    ShortestPathFollower,
)
from habitat.utils.geometry_utils import (
    direction_to_quaternion,
    quaternion_to_list,
)

GEODESIC_TO_EUCLID_RATIO_THRESHOLD = 1.1
NUMBER_RETRIES_PER_TARGET = 10000
NEAR_DIST_LIMIT = 1
FAR_DIST_LIMIT = 30
ISLAND_RADIUS_LIMIT = 1.5

DIFFICULTIES_PROP = {
    "easy": {"start": NEAR_DIST_LIMIT, "end": 6},
    "medium": {"start": 6, "end": 8},
    "hard": {"start": 8, "end": FAR_DIST_LIMIT},
}

EPISODE_COUNT = 0

TEMPLATES = {
    "find the {OBJ} in the {ROOM}{LEVEL}",
    "show me the {OBJ} in the {ROOM}{LEVEL}",
    "go to the {ROOM}{LEVEL} and find all the {OBJ} there",
    "check out the {OBJ} in the {ROOM}{LEVEL}",
}


def create_query(obj_name, obj_count, room, level=None):
    """
    :param obj_name: object category
    :param obj_count: the number of objects (not used for now)
    :param room: room name
    :param level: level id
    :return: query string
    """
    level_txt = " on level {}".format(level) if level else ""
    template = random.choice(TEMPLATES)
    txt = template.format(OBJ=obj_name, ROOM=room, LEVEL=level_txt)

    return txt


def _ratio_sample_rate(ratio):
    """
     :param ratio: geodesic distance ratio to Euclid distance
    :return: value between 0.008 and 0.144 for ration 1 and 1.1
    """
    return 20 * (ratio - 0.98) ** 2


def draw_top_down_map(info, heading, output_size):
    from habitat.utils.visualizations import maps
    import cv2

    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def create_episode(
    episode, objects, start_position, start_rotation, shortest_paths, info
) -> NavigationEpisode:
    goals = [
        ObjectGoal(
            object_id=obj["id"],
            object_name=obj["instance_id"],
            object_category=obj["category_name"],
            view_points=[
                AgentState(position=view[0][0], rotation=view[0][1])
                for view in obj["view_points"]
            ],
            position=obj["center"],
        )
        for obj in objects
    ]
    new_episode = NavigationEpisode(
        episode_id=episode.episode_id,
        goals=goals,
        scene_id=episode.scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        start_room=episode.start_room,
        shortest_paths=shortest_paths,
        info=info,
    )
    return new_episode


def get_action_shortest_path(
    env, goal_position, goal_rotation, radius=0.05, record_video=False
) -> List[ShortestPathPoint]:
    import os
    from habitat.utils.visualizations.utils import images_to_video

    # print("BEFORE:", env.sim.get_agent_state())
    # pos_bef = env.sim.get_agent_state().position.tolist()
    # rot_bef = env.sim.get_agent_state().rotation.tolist()
    observations = env.reset()
    if record_video:
        images = []
        im = observations["rgb"]
        top_down_map = draw_top_down_map(
            env.get_metrics(), observations["heading"], im.shape[0]
        )
        output_im = np.concatenate((im, top_down_map), axis=1)
        images.append(output_im)

        print("Environment creation successful")
        IMAGE_DIR = os.path.join("data", "videos")
        dirname = os.path.join(IMAGE_DIR, "eqd_v1")
    # print("EPISODE:", env.current_episode.start_position,
    #       type(env.current_episode.start_position),
    #       env.current_episode.start_rotation,
    #       type(env.current_episode.start_rotation)
    #       )
    # # env.sim.set_agent_state(env.current_episode.start_position, env.current_episode.start_rotation)
    # print("AFTER:", env.sim.get_agent_state())
    # env._task.measurements.reset_measures(episode=env.current_episode)

    mode = "greedy"

    goal_radius = radius  # env.episodes[0].goals[0].radius
    if goal_radius is None or goal_radius == 0:
        goal_radius = env._config.SIMULATOR.FORWARD_STEP_SIZE
    # scene = env.sim.semantic_annotations()
    # object_id = env.current_episode.goals[0].object_id
    # goal_center = scene.objects[object_id].obb.center.tolist()
    # env.current_episode.goals[0].position = goal_center

    follower = ShortestPathFollower(env.sim, goal_radius, False)
    follower.mode = mode

    # print("Agent stepping around inside environment.")

    shortest_path = []

    while not env.episode_over:
        action = follower.get_next_action(
            goal_position,  # env.current_episode.goals[0].position
            # goal_rotation,
            env.current_episode.goals[0].position,
        )
        state = env.sim.get_agent_state()
        # print("state.position: ", state.position, ", state.rotation: ",
        #       state.rotation)

        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action.value,
            )
        )

        observations = env.step(action.value)
        if record_video:
            im = observations["rgb"]
            top_down_map = draw_top_down_map(
                env.get_metrics(), observations["heading"], im.shape[0]
            )
            output_im = np.concatenate((im, top_down_map), axis=1)
            images.append(output_im)

    state = env.sim.get_agent_state()
    # print("state.position: ", state.position, ", state.rotation: ",
    #       state.rotation)

    shortest_path.append(
        ShortestPathPoint(
            state.position.tolist(),
            quaternion_to_list(state.rotation),
            action.value,
        )
    )

    if record_video:
        print(
            "goal_position: ",
            goal_position,
            ", goal_rotation: ",
            goal_rotation,
            "goal_rot_orig: ",
            goal_rotation,
        )
        print(
            "start_position: ",
            env.current_episode.start_position,
            ", " "start_rotation: ",
            env.current_episode.start_rotation,
        )
        video_filename = "{}_{}_{}".format(
            env.current_episode.episode_id,
            env.current_episode.goals[0].name,
            env.current_episode.goals[0].room_name,
        )
        video_filename = video_filename.replace(" ", "_")
        # print( "Heights difference: ", env.sim.get_agent_state().position[1] -
        #        goal_position[1])
        images_to_video(images, dirname, video_filename)
        print("Episode finished: {}".format(dirname + video_filename))
        # print("Shortest path length: {}".format(len(shortest_path)))

    if len(shortest_path) == env._config.ENVIRONMENT.MAX_EPISODE_STEPS:
        shortest_path = []
        print(
            "BAD SHORTEST PATH of the EPISODE: {}".format(
                env.current_episode.episode_id
            )
        )
    return [shortest_path]


def get_image_with_obj_overlay(observations, objects):
    rgb = observations["rgb"]
    sem = observations["semantic"] == -1
    for obj in objects:
        sem = np.logical_or(
            sem, observations["semantic"] == obj["instance_id"]
        )
    color_sem = np.array([(0, 0, 0), (255, 0, 0)])[np.asanyarray(sem, np.int)]
    return rgb + 0.5 * color_sem


def get_action_reference_path(
    env, objects, record_video=False
) -> List[ShortestPathPoint]:
    import os
    from habitat.utils.visualizations.utils import images_to_video
    from habitat.sims.habitat_simulator import SimulatorActions

    radius = 0.1

    # print("BEFORE:", env.sim.get_agent_state())
    # pos_bef = env.sim.get_agent_state().position.tolist()
    # rot_bef = env.sim.get_agent_state().rotation.tolist()
    observations = env.reset()
    if record_video:
        images = []
        observations = env.reset()
        im = observations["rgb"]
        top_down_map = draw_top_down_map(
            env.get_metrics(), observations["heading"], im.shape[0]
        )
        output_im = np.concatenate((im, top_down_map), axis=1)
        images.append(output_im)
        IMAGE_DIR = os.path.join("data", "videos")
        dirname = os.path.join(IMAGE_DIR, "eqd_v1")

    # print("EPISODE:", env.current_episode.start_position,
    #       type(env.current_episode.start_position),
    #       env.current_episode.start_rotation,
    #       type(env.current_episode.start_rotation)
    #       )
    # # env.sim.set_agent_state(env.current_episode.start_position, env.current_episode.start_rotation)
    # print("AFTER:", env.sim.get_agent_state())
    # env._task.measurements.reset_measures(episode=env.current_episode)

    mode = "greedy"
    goal_radius = radius  # env.episodes[0].goals[0].radius
    if goal_radius is None or goal_radius == 0:
        goal_radius = env._config.SIMULATOR.FORWARD_STEP_SIZE
    # scene = env.sim.semantic_annotations()
    # object_id = env.current_episode.goals[0].object_id
    # goal_center = scene.objects[object_id].obb.center.tolist()
    # env.current_episode.goals[0].position = goal_center

    follower = ShortestPathFollower(env.sim, goal_radius, False)
    follower.mode = mode

    shortest_path = []
    geo_dist_time = 0
    objects_to_visit = objects.copy()
    while len(objects_to_visit) > 0:

        cur_position = env.sim.get_agent_state().position
        view_points = [
            (obj_id, view)
            for obj_id, obj in enumerate(objects_to_visit)
            for view in obj["view_points"]
        ]

        # view_dist = np.array(list(map(lambda o: [env.sim.geodesic_distance(
        #     cur_position, o[1][0][0]), o[1][0]], view_points)))

        view_dist = np.asarray(
            list(
                map(
                    lambda o: env.sim.geodesic_distance(
                        cur_position, o[1][0][0]
                    ),
                    view_points,
                )
            )
        )
        next_view = view_points[np.argmin(view_dist, axis=0)]
        goal_position = next_view[1][0][0]
        goal_rotation = next_view[1][0][1]
        del objects_to_visit[next_view[0]]
        action = None

        while (
            action != SimulatorActions.STOP
            and len(shortest_path) <= env._config.ENVIRONMENT.MAX_EPISODE_STEPS
        ):
            t_geo_dist_time = time()
            action = follower.get_next_action(
                goal_position,  # env.current_episode.goals[0].position
                goal_rotation,
                # env.current_episode.goals[0].position
            )
            geo_dist_time += time() - t_geo_dist_time
            state = env.sim.get_agent_state()
            # print("state.position: ", state.position, ", state.rotation: ",
            #       state.rotation)

            shortest_path.append(
                ShortestPathPoint(
                    state.position.tolist(),
                    quaternion_to_list(state.rotation),
                    action.value,
                )
            )

            if action == SimulatorActions.STOP:
                # print("Target #{} found.".format(len(objects_to_visit) + 1))
                break
            observations = env.step(action.value)
            env.sim._is_episode_active = True
            env._episode_over = False

            if record_video:
                im = get_image_with_obj_overlay(observations, objects)
                top_down_map = draw_top_down_map(
                    env.get_metrics(), observations["heading"], im.shape[0]
                )
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)

    state = env.sim.get_agent_state()
    # print("state.position: ", state.position, ", state.rotation: ",
    #       state.rotation)
    if action is not None:
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action.value,
            )
        )

    # print("goal_position: ", goal_position, ", goal_rotation: ",
    #       goal_rotation, "goal_rot_orig: ", goal_rotation)
    # print("start_position: ", env.current_episode.start_position, ", "
    #                                                               "start_rotation: ",
    #       env.current_episode.start_rotation)
    print(f"shortest_follower_time: {geo_dist_time}")
    if record_video:
        video_filename = "{}_{}_{}".format(
            env.current_episode.episode_id,
            env.current_episode.goals[0].name,
            env.current_episode.goals[0].room_name,
        )
        video_filename = video_filename.replace(" ", "_")
        # print( "Heights difference: ", env.sim.get_agent_state().position[1] -
        #        goal_position[1])
        images_to_video(images, dirname, video_filename)
        print("Episode finished: {}".format(dirname + video_filename))

    if len(shortest_path) >= env._config.ENVIRONMENT.MAX_EPISODE_STEPS:
        shortest_path = []
        print(
            "BAD SHORTEST PATH of the EPISODE: {}".format(
                env.current_episode.episode_id
            )
        )
    return [shortest_path]


# ShortestPathPoint(position, rotation, action)
#                      for position, rotation, action in
#                      zip(action_shortest_path.points,
#                          action_shortest_path.rotations,
#                          actions)


def generate_new_start_eqa_episode(env, num_episodes=-1):
    episode = None
    episode_count = 0
    # while True: #not episode:

    target_position = env.current_episode.shortest_paths[0][-1].position
    target_rotation = habitat.utils.geometry_utils.quaternion_xyzw_to_wxyz(
        env.current_episode.shortest_paths[0][-1].rotation
    )
    if (
        env._sim._sim.pathfinder.island_radius(target_position)
        < ISLAND_RADIUS_LIMIT
    ):
        print("ISLAND_RADIUS_LIMIT")
        yield None

    if not env._sim.is_navigable(target_position):
        print("NOT NAVIGABLE END")
        yield None
    while episode_count < num_episodes or num_episodes < 0:
        # target_position =
        # env._sim._sim.pathfinder.get_random_navigable_point()
        # habitat.utils.geometry_utils.quaternion_xyzw_to_wxyz(env.current_episode.shortest_paths[0][-1].rotation)
        # quaternion_wxyz_to_xyzw
        # print("TARGET radius: ", env._sim._sim.pathfinder.island_radius(
        #    target_position), target_position[1])
        if (
            env._sim._sim.pathfinder.island_radius(target_position)
            < ISLAND_RADIUS_LIMIT
        ):
            # print("ISLAND_RADIUS_LIMIT")
            continue

        for retry in range(NUMBER_RETRIES_PER_TARGET):
            source_position = (
                env._sim._sim.pathfinder.get_random_navigable_point()
            )
            # print("source radius: ", env._sim._sim.pathfinder.island_radius(
            #    source_position), source_position[1])
            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                env,
                near_dist=DIFFICULTIES_PROP["easy"]["start"],
                far_dist=DIFFICULTIES_PROP["hard"]["end"],
            )
            if is_compatible:
                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
                env.sim.set_agent_state(source_position, source_rotation)
                env.current_episode.goals[0].position = target_position
                env.current_episode.start_position = source_position.tolist()
                env.current_episode.start_rotation = source_rotation

                # scene = env.sim.semantic_annotations()
                # object_id = env.current_episode.goals[0].object_id
                # target_position = scene.objects[object_id].obb.center

                shortest_paths = get_action_shortest_path(
                    env,
                    goal_position=target_position,
                    goal_rotation=target_rotation,
                    # env.current_episode.goals[0].position #target_position
                )
                # logger.info("Geo dist: {}".format(dist))
                # print("SUCCESS TARGET radius: ",
                #       env._sim._sim.pathfinder.island_radius(
                #           target_position), target_position[1])
                # print("SUCCESS source radius: ",
                #       env._sim._sim.pathfinder.island_radius(
                #           source_position), source_position[1])

                episode = create_episode(
                    episode=env.current_episode,
                    start_position=source_position.tolist(),
                    start_rotation=source_rotation,
                    # target_position=target_position,#.tolist(),
                    shortest_paths=shortest_paths,
                    info={"geodesic_distance": dist},
                )

                # make_greedy_path_video(episode, env,
                #                 "data/videos/mp3d/debug/" if "mp3d" in
                #                                             env._sim.config.SCENE else "data/videos/gibson/debug/"
                #                 )
                episode_count += 1
                # print("is compatible")
                yield episode
                break
    # return episode


def compute_iou(cand_mask, ref_mask=None):
    """
    Given (h, w) cand_mask, we wanna our ref_mask to be in the center of image,
    with [0.25h:0.75h, 0.25w:0.75w] occupied.
    """
    if ref_mask is None:
        h, w = cand_mask.shape[0], cand_mask.shape[1]
        ref_mask = np.zeros((h, w), np.int8)
        ref_mask[
            int(0.25 * h) : int(0.85 * h), int(0.25 * w) : int(0.75 * w)
        ] = 1

    inter = (cand_mask > 0) & (ref_mask > 0)
    union = (cand_mask > 0) | (ref_mask > 0)
    iou = inter.sum() / (union.sum())
    # print("compute_iou: {}".format(iou))
    return iou


def get_view_state_object(sim, target_centroid):
    step_size = habitat.get_config().SIMULATOR.FORWARD_STEP_SIZE
    width = 2.0
    xv, yv, zv = np.meshgrid(
        np.linspace(-width, width, num=int(2.0 * width / step_size))
        + target_centroid[0],
        np.linspace(-1, 0, num=int(1.0 * width / step_size))
        + target_centroid[1],
        np.linspace(-width, width, num=int(2.0 * width / step_size))
        + target_centroid[2],
    )
    for x, y, z in zip(xv.flatten(), yv.flatten(), zv.flatten()):
        candidate_point = [x, y, z]  # target_centroid[1] - 0.6,
        if sim.is_navigable(candidate_point):
            return candidate_point
    return None


def get_view_states(sim: habitat.Simulator, obj, target_centroid):
    step_size = habitat.get_config().SIMULATOR.FORWARD_STEP_SIZE
    turn_angle = 2 * np.pi * habitat.get_config().SIMULATOR.TURN_ANGLE / 360

    width = 2.0
    xv, yv, zv = np.meshgrid(
        np.linspace(-width, width, num=int(2.0 * width / step_size))
        + target_centroid[0],
        np.linspace(-1, 0, num=20) + target_centroid[1],
        np.linspace(-width, width, num=int(2.0 * width / step_size))
        + target_centroid[2],
    )

    thetas = np.linspace(
        0.0, 2.0 * np.pi - turn_angle, num=int(2.0 * np.pi / turn_angle)
    )
    headings = [[0, np.sin(t / 2), 0, np.cos(t / 2)] for t in thetas]

    def count_pixels(pos):
        goal_direction = np.array(target_centroid) - np.array(pos)
        goal_direction[1] = 0
        # goal_rot = mp3d_direction_to_hsim_quaternion(goal_direction)
        goal_rot = direction_to_quaternion(goal_direction)

        def _iou(heading):
            sim.set_agent_state(pos, heading)
            sim_obs = sim._sim.get_sensor_observations()
            obs = sim._sensor_suite.get_observations(sim_obs)

            result = (
                (pos, heading),
                compute_iou((obs["semantic"] == obj["instance_id"])),
            )
            # if (result[1] > 0):
            #     imageio.imsave(
            #         os.path.join(
            #             "/private/home/maksymets/hapi_eqd/data/images/",
            #             "{}_{}_target_image.png".format(obj[
            #                                                 'category_name'],
            #                                                result[1]
            #             )),
            #         obs["rgb"]
            #     )
            return result

        # object_id = int(query[1]["objects"][0]["id"].split("_")[-1])
        #     retun (py_().map(compute_iou)(objs))

        return (
            py_()
            .map(_iou)
            .max_by(1)([goal_rot])
            #                (headings[np.random.randint(0, 1)::4])
        )

    def sample_positions(positions):
        inds = np.arange(len(positions))
        max_samples = int(4e3)
        if len(positions) > max_samples:
            inds = np.random.choice(inds, size=max_samples, replace=False)

        return [positions[i] for i in inds]

    # py_().filter(lambda pos: sim.is_navigable(pos)).thru(sample_positions).map(
    #     count_pixels).filter(lambda v: v[1] > 0.075).sort_by(1,
    #                                                          reverse=True).take_while(
    #     lambda v, idx, arr: (v[1] / (arr[0][1] + 1e-10)) > 0.85).map(0).map(
    #     lambda hp: (hp[0], hp[1]))(
    #     zip(xv.flatten(), yv.flatten(), zv.flatten()))

    return (
        py_()
        # .map(lambda xy: np.array([xy[0], xy[2], target_centroid[1]]))
        .map(lambda pos: np.array(pos))
        .filter(lambda pos: sim.is_navigable(pos))
        .filter(
            lambda pos: sim._sim.pathfinder.island_radius(pos)
            > ISLAND_RADIUS_LIMIT
        )
        # .thru(sample_positions)
        .map(count_pixels)
        .filter(lambda v: v[1] > 0.075)
        .take_while(lambda v, idx, arr: (v[1] / (arr[0][1] + 1e-10)) > 0.85)
        .sort_by(1, reverse=True)
        #            .map(0)
        #            .map(lambda hp: (hp[0].tolist(), hp[1].tolist(), hp[2].tolist()))
        (zip(xv.flatten(), yv.flatten(), zv.flatten()))
    )


def generate_new_obj_nav_episode(
    env,
    # sim: Simulator,
    objects,
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
) -> NavigationEpisode:
    env.current_episode.episode_id = 0
    print("Number of objects: ", len(objects))
    for obj in objects:
        if isinstance(obj["id"], str):
            obj["instance_id"] = int(obj["id"].split("_")[-1])
        else:
            obj["instance_id"] = obj["id"]
        # Single navigable point around.
        # target_position = get_view_state_object(env.sim, objects[0]["center"])
        view_points = get_view_states(env.sim, obj, obj["center"])
        if not view_points:
            print("Failed find position for {}".format(obj["category_name"]))
            return
            yield

        print(
            "Number of views: ",
            len(view_points),
            " for ",
            obj["category_name"],
        )
        # target_position = view_points[0][0][0]
        # target_rotation = view_points[0][0][1]
        obj["view_points"] = view_points
        # target_position = env.current_episode.shortest_paths[0][-1].position

    shortest_paths = []

    env.current_episode.goals[0].position = objects[0]["center"]

    episode_count = 0
    gen_shortest_path = 0
    while episode_count < num_episodes or num_episodes < 0:
        for retry in range(number_retries_per_target):
            source_position = env.sim.sample_navigable_point()

            comp_results = list(
                map(
                    lambda obj: is_compatible_episode(
                        source_position,
                        obj["view_points"][0][0][0],
                        env.sim,
                        near_dist=closest_dist_limit,
                        far_dist=furthest_dist_limit,
                        geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                    ),
                    objects,
                )
            )
            from functools import reduce

            all_compatible = reduce(
                lambda acum, el: acum and el[0], comp_results
            )
            # closest_obj = np.argmin(np.array(list(map(lambda x:x[1], comp_results))))

        if all_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            env.current_episode.start_position = source_position
            env.current_episode.start_rotation = source_rotation

            env.current_episode.goals[0].name = objects[0]["category_name"]
            t_gen_shortest_path = time()
            cur_shortest_paths = get_action_reference_path(env, objects)
            print(f"t_gen_shortest_path: {time() - t_gen_shortest_path}")

            if cur_shortest_paths and len(cur_shortest_paths) > 0:
                shortest_paths.append(cur_shortest_paths[0])

            episode = create_episode(
                episode=env.current_episode,
                query=query,
                start_position=env.current_episode.start_position,
                start_rotation=env.current_episode.start_rotation,
                shortest_paths=shortest_paths,
                info={
                    # "geodesic_distance": env._sim.geodesic_distance(env.current_episode.start_position, target_position),
                },
            )

            episode_count += 1
            yield episode
