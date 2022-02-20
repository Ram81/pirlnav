#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import gzip
import habitat
import json
import os
import random
import scipy
import sys
import habitat_sim
import itertools
import math

import numpy as np
import magnum as mn
import matplotlib.pyplot as plt

from collections import defaultdict
from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_coeffs, quat_from_magnum, quat_to_coeffs
from habitat_sim.geo import OBB
from mpl_toolkits.mplot3d import Axes3D


ISLAND_RADIUS_LIMIT = 1.5
VISITED_POINT_DICT = {}
object_to_rooms_map = defaultdict(list)
region_bb_map = {}
points_in_room = []
region_point_map = {}


def get_object_handle(object_name):
    return "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)


def contact_test(sim, object_name, position):
    object_handle = "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)
    return sim.pre_add_contact_test(object_handle, mn.Vector3(position))


def contact_test_rotation(sim, object_name, position, rotation):
    object_handle = "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)
    return sim.pre_add_contact_test(object_handle, mn.Vector3(position), quat_from_coeffs(rotation))


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.
    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s, t, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation


def get_n_objects_groups(objects, k):
    groups = []
    for i in range(0, len(objects), k):
        group = objects[i : i + k].copy()
        group.sort()
        if len(group) == 4:
            groups.append(group)
    return groups


def get_all_n_object_groups(objects, k=4, n=100):
    object_groups = []
    visited_groups = defaultdict(int)
    shuffled_objects = objects
    while len(object_groups) < n:
        random.shuffle(objects)
        groups = get_n_objects_groups(shuffled_objects, k)
        for group in groups:
            group_str = "_".join(group)
            if visited_groups[group_str] == 0:
                object_groups.append(group)
                visited_groups[group_str] += 1
    return object_groups


def populate_objects_to_rooms_map(room_to_objects_map):
    for room, objects in room_to_objects_map.items():
        for object_ in objects:
            object_to_rooms_map[object_].append(room)


def get_room_object_groups(room, k=4, n=100):
    filtered_objects = []
    for object_, rooms in object_to_rooms_map.items():
        if room not in rooms:
            filtered_objects.append(object_)

    object_groups = get_all_n_object_groups(filtered_objects, k, n)
    return object_groups


def get_object_receptacle_pair(object_to_receptacle_list, index):
    index = index % len(object_to_receptacle_list)
    return object_to_receptacle_list[index]


def use_in_for_receptacle(receptacle_name):
    for part in ["bowl", "cube", "bin", "basket"]:
        if part in receptacle_name.lower():
            return True
    return False


def get_task_config(config, room_name, objects):
    task = {}
    task["instruction"] = config["TASK"]["INSTRUCTION"].format(room_name)
    task["type"] = config["TASK"]["TYPE"]
    task["goals"] = {}

    object_to_room_map = defaultdict(list)
    for object_ in objects:
        object_id = object_["objectId"]
        object_name = object_["objectHandle"].split("/")[-1].split(".")[0]
        object_to_room_map[object_id] = object_to_rooms_map.get(object_name)

    task["goals"]["objectToRoomMap"] = object_to_room_map
    return task


def build_episode(config, episode_id, objects, agent_position, agent_rotation, room_name):
    scene_id = config.SIMULATOR.SCENE.split("/")[-1]
    task_config = config.TASK
    episode = {}
    episode["episode_id"] = episode_id
    episode["scene_id"] = scene_id
    episode["start_position"] = agent_position
    episode["start_rotation"] = agent_rotation

    object_ids = []
    receptacle_ids = []
    for object_ in objects:
        if not object_["isReceptacle"]:
            object_ids.append(object_["objectId"])
        else:
            receptacle_ids.append(object_["objectId"])
    
    episode["task"] = get_task_config(config, room_name, objects)
    episode["objects"] = objects
    return episode



def build_object(object_handle, object_id, object_name, is_receptacle, position, rotation):
    object_ = {
        "object": object_name,
        "objectHandle": "/data/objects/{}.object_config.json".format(object_handle),
        "objectIcon": "/data/test_assets/objects/{}.png".format(object_handle),
        "objectId": object_id,
        "isReceptacle": is_receptacle,
        "position": position,
        "rotation": rotation,
        "motionType": "DYNAMIC"
    }
    return object_


def get_bad_points(
    sim, points, rotations, d_lower_lim, d_upper_lim,
    geodesic_to_euclid_min_ratio, xlim=None,
    ylim=None, zlim=None, is_tilted_or_colliding=[], room_bb=None
):
    bad_points = np.zeros(points.shape[0], dtype=bool)
    # Outside X, Y, or Z limits
    if xlim:
        bad_points[points[:, 0] < xlim[0]] = 1
        bad_points[points[:, 0] > xlim[1]] = 1

    if ylim:
        bad_points[points[:, 2] < ylim[0]] = 1
        bad_points[points[:, 2] > ylim[1]] = 1

    if zlim:
        bad_points[points[:, 1] < zlim[0]] = 1
        bad_points[points[:, 1] > zlim[1]] = 1

    for i, point in enumerate(points):
        point_list = point.tolist()
        existing_point_count = VISITED_POINT_DICT.get(str(point_list))
        if existing_point_count is not None and existing_point_count >= 1: # or is_tilted_or_colliding[i]:
            bad_points[i] = 1

    # Too close to another object or receptacle
    point1 = points[0]

    for j, point2 in enumerate(points):
        if j == 0:
            continue

        is_compatible, dist = is_compatible_episode(
            point1,
            point2,
            sim,
            near_dist=d_lower_lim,
            far_dist=d_upper_lim,
            geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
        )

        if not is_compatible:
            bad_points[j] = 1

    return bad_points


def rejection_sampling(
    sim, points, rotations, d_lower_lim, d_upper_lim,
    geodesic_to_euclid_min_ratio, xlim=None,
    ylim=None, zlim=None, num_tries=10000, object_names=[], room_bb=None,
    is_tilted_or_colliding=[]
):
    bad_points = get_bad_points(
        sim, points, rotations, d_lower_lim, d_upper_lim,
        geodesic_to_euclid_min_ratio, xlim, ylim,
        zlim, is_tilted_or_colliding, room_bb=room_bb
    )

    while sum(bad_points) > 0 and num_tries > 0:

        for i, bad_point in enumerate(bad_points):
            if bad_point and i == 0:
                points[i] = get_random_point(sim)
            elif bad_point:
                points[i], rotations[i], is_tilted_or_colliding[i] = get_random_object_position(
                    sim, object_names[i - 1], room_bb=room_bb
                )

        bad_points = get_bad_points(
            sim, points, rotations, d_lower_lim, d_upper_lim,
            geodesic_to_euclid_min_ratio, xlim, ylim,
            zlim, is_tilted_or_colliding, room_bb=room_bb
        )
        num_tries -= 1
    
    print(sum(bad_points), sum(is_tilted_or_colliding), num_tries)

    if sum(bad_points) > 0:
        print("\n Error generating unique points, try using bigger retries")
        sys.exit(1)

    return points


def rejection_sampling_no_objects(
    sim, points, rotations, d_lower_lim, d_upper_lim,
    geodesic_to_euclid_min_ratio, xlim=None,
    ylim=None, zlim=None, num_tries=10000, room_bb=None
):
    bad_points = get_bad_points(
        sim, points, rotations, d_lower_lim, d_upper_lim,
        geodesic_to_euclid_min_ratio, xlim, ylim,
        zlim, room_bb=room_bb
    )
    largest_object = "Room_Essentials_Fabric_Cube_Lavender"
    while sum(bad_points) > 0 and num_tries > 0:

        for i, bad_point in enumerate(bad_points):
            # points[i] = get_random_point(sim, room_bb)
            points[i], _, tilt = get_random_object_position(sim, largest_object, room_bb)

        bad_points = get_bad_points(
            sim, points, rotations, d_lower_lim, d_upper_lim,
            geodesic_to_euclid_min_ratio, xlim, ylim,
            zlim, room_bb
        )
        num_tries -= 1
    
    print(sum(bad_points), num_tries)

    if sum(bad_points) > 0:
        print("\n Error generating unique points, try using bigger retries")
        # sys.exit(1)

    return points


def get_random_point(sim, bb=None):
    point = np.array([0, 0, 0])
    if bb:
        bb_point = np.random.uniform(bb.min, bb.max)
        snapped_point = sim.pathfinder.snap_point(bb_point)
        while np.isnan(snapped_point[0]):
            bb_point = np.random.uniform(bb.min, bb.max)
            snapped_point = sim.pathfinder.snap_point(bb_point)
        point = np.array(snapped_point).tolist()
    else:
        point = sim.sample_navigable_point()
    return point


def get_random_rotation():
    angle = np.random.uniform(0, 2 * np.pi)
    rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    return rotation


def is_less_than_island_radius_limit(sim, point):
    return sim.island_radius(point) < ISLAND_RADIUS_LIMIT


def add_contact_test_object(sim, object_name):
    object_handle = get_object_handle(object_name)
    sim.add_contact_test_object(object_handle)


def remove_contact_test_object(sim, object_name):
    object_handle = get_object_handle(object_name)
    sim.remove_contact_test_object(object_handle)


def step_physics_n_times(sim, n=10, dt = 1.0 / 10.0):
    sim.step_world(n * dt)


def is_num_active_collision_points_zero(sim):
    return (sim.get_num_active_contact_points() == 0)


def test_contact_on_settled_point(sim, object_name, position):
    object_handle = get_object_handle(object_name)
    temp_pos = mn.Vector3(position[0], position[1] + 0.2, position[2])
    return contact_test(sim, object_name, temp_pos)


def get_random_object_position(sim, object_name, room_bb=None, scene_collision_margin=0.04, tilt_threshold=0.95):
    object_handle = get_object_handle(object_name)
    position = get_random_point(sim, room_bb)

    if position[0] == math.inf:
        translation = np.array([0, 0, 0])
        rotation = np.array([0, 0, 0, 1])
        is_tilted_or_colliding = True
        print("Got inf point: {}".format(position))
        return translation, rotation, is_tilted_or_colliding

    object_id  = sim.add_object_by_handle(object_handle)
    obj_node = sim.get_object_scene_node(object_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )
    # also account for collision margin of the scene
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    position = mn.Vector3(position) + y_translation
    sim.set_translation(position, object_id)
    is_colliding = sim.contact_test(object_id)

    # If colliding that means object near a wall or on the floor
    is_physics_stepped = False
    if not is_colliding:
        step_physics_n_times(sim)
        is_physics_stepped = True

    translation = sim.get_translation(object_id)
    rotation = sim.get_rotation(object_id)

    orientation = sim.get_rotation(object_id)
    object_up = orientation.transform_vector(mn.Vector3(0,1,0))
    tilt = mn.math.dot(object_up, mn.Vector3(0,1,0))
    is_tilted = (tilt <= tilt_threshold)

    if is_physics_stepped:
        adjusted_translation = mn.Vector3(
            0, scene_collision_margin, 0
        ) + translation
        sim.set_translation(adjusted_translation, object_id)
        is_colliding = sim.contact_test(object_id)

    query = np.array(translation)
    is_tilted_or_colliding = (is_tilted or is_colliding)

    rotation = quat_to_coeffs(quat_from_magnum(rotation)).tolist()
    translation = np.array(translation).tolist()
    sim.remove_object(object_id)
    return np.array(translation), rotation, is_tilted_or_colliding


def populate_episodes_points(episodes, scene_id, task_type):
    for episode in episodes["episodes"]:
        if scene_id != episode["scene_id"] or episode["task"]["type"] != task_type:
            continue
        point = str(episode["start_position"])
        if VISITED_POINT_DICT.get(point):
            VISITED_POINT_DICT[point] += 1
            print("Redundant agent position in episode {}".format(episode["episode_id"]))
        else:
            VISITED_POINT_DICT[point] = 1

        for object_ in episode["objects"]:
            point = str(object_["position"])
            if VISITED_POINT_DICT.get(point):
                VISITED_POINT_DICT[point] += 1
                print("Redundant point in episode {}".format(episode["episode_id"]))
            else:
                VISITED_POINT_DICT[point] = 1


def populate_prev_generated_points(path, scene_id, task_type):
    for file_path in glob.glob(path + "/*.json"):
        with open(file_path, "r") as file:
            data = json.loads(file.read())
            populate_episodes_points(data, scene_id, task_type)
    print("Total previously generated points {}".format(len(VISITED_POINT_DICT.keys())))


def remove_all_objects(sim):
    for object_id in sim.get_existing_object_ids():
        sim.remove_object(object_id)
    sim.clear_recycled_object_ids()


def sample_n_points_in_room(
    sim, room_bb, d_lower_lim, d_upper_lim, geodesic_to_euclid_min_ratio, number_retries_per_target, n=700
):
    points = []
    rotations = []
    room_obb = OBB(room_bb)
    largest_object = "Room_Essentials_Fabric_Cube_Lavender"
    for i in range(n):
        point, _, tilt = get_random_object_position(sim, largest_object, room_bb)
        # if room_bb.contains(point, 1e-6):
        #     points.append(point)
        in_obb = room_obb.contains(point, 1e-6)
        if in_obb and not tilt:
            points.append(point.tolist())
    
    print("Found {} points in 1 trial".format(len(points)))
    # agent points
    # for i in range(number_retries_per_target):
    #     point, _, tilt = get_random_object_position(sim, largest_object, room_bb)
    #     # if room_bb.contains(point, 1e-6):
    #     #     points.append(point)
    #     in_obb = room_obb.contains(point, 1e-6)
    #     if in_obb and not tilt:
    #         points.append(point)
    #     if len(points) == n:
    #         break
        
    print("Found {} points in {} trials".format(len(points), i))
    if len(points) > 0:
        dist = scipy.spatial.distance.pdist(points)
        print("Min dist between points: {} -- {}".format(np.min(dist), np.sum(dist >= 0.2)))
    return points


def populate_region_bb(sim, room_to_objects_map, d_lower_lim, d_upper_lim, geodesic_to_euclid_min_ratio, number_retries_per_target):
    levels = []
    semantic_scene = sim.semantic_scene
    rooms = room_to_objects_map.keys()
    for level in semantic_scene.levels:
        level_id = int(level.id)
        region_bb_map[level_id] = {}
        region_point_map[level_id] = {}
        for region in level.regions:
            region_name = region.category.name()
            if region_name not in rooms:
                continue
            print(level_id, region_name)
            if region_name not in region_bb_map[level_id].keys():
                region_bb_map[level_id][region_name] = []
                region_point_map[level_id][region_name] = []
            obb = OBB(region.aabb)
            points = sample_n_points_in_room(sim, region.aabb, d_lower_lim, d_upper_lim, geodesic_to_euclid_min_ratio, number_retries_per_target)
            print(region.aabb.min, region.aabb.max)
            print("Sampled {} points at distance {} - {} for room {}".format(len(points), d_lower_lim, d_upper_lim, region_name))
            region_bb_map[level_id][region_name].append(region.aabb)
            region_point_map[level_id][region_name].extend(points)
        levels.append(level_id)
    
    write_json(region_point_map, "room_points")
    return levels


def generate_tasks(
    num_targets,
    num_objects_per_task
):
    room_to_object_groups = {}
    room_to_objects_map = config["TASK"]["ROOM_OBJECTS_MAP"]

    for room in room_to_objects_map.keys():
        object_groups = get_room_object_groups(room, num_objects_per_task, num_targets)
        room_to_object_groups[room] = object_groups

    write_json(room_to_object_groups, "tasks")



def generate_points(
    config,
    objs_per_rec,
    num_episodes,
    num_targets,
    number_retries_per_target=1000,
    d_lower_lim=0.2,
    d_upper_lim=30.0,
    geodesic_to_euclid_min_ratio=1.1,
    prev_episodes="data/tasks",
    scene_id="empty_house.glb",
):
    # Initialize simulator
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    # Populate previously generated points
    populate_prev_generated_points(prev_episodes, scene_id, config["TASK"]["TYPE"])

    episode_count = 0
    episodes = []
    num_objects_per_task = config["TASK"]["NUM_OBJECTS"]
    
    room_to_objects_map = config["TASK"]["ROOM_OBJECTS_MAP"]
    populate_objects_to_rooms_map(room_to_objects_map)
    levels = populate_region_bb(sim, room_to_objects_map, d_lower_lim, d_upper_lim, geodesic_to_euclid_min_ratio, number_retries_per_target)
    object_name_map = dict(config["TASK"]["OBJECT_NAME_MAP"])
    y_limit = config["TASK"].get("Y_LIMIT")
    x_limit = config["TASK"].get("X_LIMIT")

    num_points = num_objects_per_task + 1

    task_file = config["TASK"]["TASK_MAP_FILE"]
    room_to_object_groups = json.loads(open(task_file, "r").read())

    num_tasks = 0
    for room, object_groups in room_to_object_groups.items():
        num_tasks += len(object_groups)

    num_episodes = num_tasks * num_episodes
    print("Generating total {} episodes for {} rooms".format(num_episodes, len(room_to_objects_map.keys())))
    while episode_count < num_episodes:
        for room, object_groups in room_to_object_groups.items():
            print("Generating {} episodes for room {}".format(len(object_groups), room))
            all_points = region_point_map[0][room]
            for i, object_list in enumerate(object_groups):
                print("\nEpisode {}\n".format(episode_count))
                lb = i * 5
                ub = (i + 1) * 5
                objects = []
                object_names = []
                for object_ in object_list:
                    object_names.append(object_name_map[object_])

                index = 0 
                print(levels, index)
                level = levels[index]
                room_bb = region_bb_map[level][room][0]

                points = all_points[lb:ub]
                rotations = []
                is_tilted_or_colliding = []
                num_points = len(object_list) + 1
                for idx in range(num_points):
                    rotation = get_random_rotation()
                    rotations.append(rotation)
                
                points = np.array(points)
                print(points.shape)
                points = rejection_sampling_no_objects(
                    sim, points, rotations, d_lower_lim, d_upper_lim,
                    geodesic_to_euclid_min_ratio, xlim=x_limit, ylim=y_limit,
                    num_tries=number_retries_per_target/100, room_bb=room_bb
                )

                # Mark valid points as visited to get unique points
                print("Total unique points: {}".format(len(VISITED_POINT_DICT.keys())))
                for i, point in enumerate(points):
                    VISITED_POINT_DICT[str(point.tolist())] = 1
                    # Create episode object configs
                    if i != 0:
                        objects.append(build_object(object_list[i-1], len(objects), object_names[i-1], False, points[i].tolist(), rotations[i]))

                agent_position = points[0].tolist()
                agent_rotation = rotations[0]

                # Build episode from object and agent initilization.
                episode = build_episode(
                    config, episode_count, objects, agent_position,
                    agent_rotation, room
                )
                episodes.append(episode)

                remove_all_objects(sim)
                episode_count += 1
                if episode_count == 10:
                    break
            break
        break


    dataset = {
        "episodes": episodes
    }
    return dataset


def write_json(data, file_name):
    path = "data/points/{}.json".format(file_name)
    with open(path, "w") as f:
        f.write(json.dumps(data))


def write_episode(dataset, filename):
    prefix = "data/tasks/" + filename
    with open(prefix, "w") as output_file:
        output_file.write(json.dumps(dataset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate new episodes."
    )
    parser.add_argument(
        "--task-config",
        default="psiturk_dataset/rearrangement_task_2.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--scenes",
        help="Scenes"
    )
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to generate per object receptacle pair",
    )
    parser.add_argument(
        "-g",
        "--num_targets",
        type=int,
        default=10,
        help="Number of target per room to sample",
    )
    parser.add_argument(
        "--number_retries_per_target",
        type=int,
        default=10,
        help="Number of retries for each target",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="episode_1.json",
        help="Output file for episodes",
    )
    parser.add_argument(
        "--d_lower_lim",
        type=float,
        default=0.2,
        help="Closest distance between objects allowed.",
    )
    parser.add_argument(
        "--d_upper_lim",
        type=float,
        default=30.0,
        help="Farthest distance between objects allowed.",
    )
    parser.add_argument(
        "--geodesic_to_euclid_min_ratio",
        type=float,
        default=1.1,
        help="Geodesic shortest path to Euclid distance ratio upper limit till aggressive sampling is applied.",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=4,
        help="Number of objects per goal.",
    )
    parser.add_argument(
        "--prev_episodes",
        default="data/tasks",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--gen-task", dest='gen_tasks', action='store_true'
    )

    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)

    dataset_type = config.DATASET.TYPE
    scene_id = ""
    if args.scenes is not None:
        config.defrost()
        config.SIMULATOR.SCENE = args.scenes
        config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
        config.freeze()
        scene_id = args.scenes.split("/")[-1]

    if dataset_type == "Interactive":
        if args.gen_tasks:
            generate_tasks(
                args.num_targets,
                args.ratio,
            )
        else:                
            dataset = generate_points(
                config,
                args.ratio,
                args.num_episodes,
                args.num_targets,
                args.number_retries_per_target,
                args.d_lower_lim,
                args.d_upper_lim,
                args.geodesic_to_euclid_min_ratio,
                args.prev_episodes,
                scene_id,
            )
            write_episode(dataset, args.output)
    else:
        print(f"Unknown dataset type: {dataset_type}")