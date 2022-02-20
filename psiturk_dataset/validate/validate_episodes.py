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
import math

import numpy as np
import magnum as mn
import matplotlib.pyplot as plt
import scipy

from collections import defaultdict
from habitat_sim.nav import NavMeshSettings
from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_coeffs, quat_from_magnum, quat_to_coeffs, quat_to_magnum
from mpl_toolkits.mplot3d import Axes3D
from psiturk_dataset.utils.utils import load_dataset


ISLAND_RADIUS_LIMIT = 1.5
VISITED_POINT_DICT = {}


def get_geodesic_distance(sim, position_a, position_b):
    return sim.geodesic_distance(position_a, position_b)


def get_object_handle(object_name):
    return "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)


def add_contact_test_object(sim, object_name):
    object_handle = get_object_handle(object_name)
    sim.add_contact_test_object(object_handle)


def add_object(sim, object_name):
    object_handle = get_object_handle(object_name)
    return sim.add_object_by_handle(object_handle)


def contact_test_rotation(sim, object_name, position, rotation):
    object_handle = "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)
    return sim.pre_add_contact_test(object_handle, mn.Vector3(position), quat_from_coeffs(rotation))


def contact_test(sim, object_name, position):
    object_handle = "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)
    return sim.pre_add_contact_test(object_handle, mn.Vector3(position))


def get_object_handle(object_name):
    return "data/scene_datasets/habitat-test-scenes/../../test_assets/objects/{}.object_config.json".format(object_name)


def are_points_navigable(sim, episode):
    pathfinder = sim.pathfinder
    agent_position = episode["start_position"]

    positions = [agent_position]

    is_navigable = pathfinder.is_navigable(agent_position)
    is_navigable_list = [is_navigable]
    for object_ in episode["objects"]:
        object_name = object_["objectHandle"].split("/")[-1].split(".")[0]
        position = object_["position"]
        is_navigable = pathfinder.is_navigable(position)
        is_navigable_list.append(is_navigable)
        positions.append(position)
    
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i <= j:
                continue
            dist = get_geodesic_distance(sim, positions[i], positions[j])
            if dist == np.inf or dist == math.inf:
                return False
    
    if np.sum(is_navigable_list) != 3:
        return False
    return True


def get_points_distance(points, sim, show_plot=False):
    print("Min distance between points...")
    agent_idxs = [0]
    object_idxs = [1]
    receptacle_idxs = [2]

    for i in range(int(len(points)/3) - 1):
        agent_idxs.append(agent_idxs[-1] + 3)
        object_idxs.append(object_idxs[-1] + 3)
        receptacle_idxs.append(receptacle_idxs[-1] + 3)

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            points[agent_idxs][:, 0],
            points[agent_idxs][:, 2],
            points[agent_idxs][:, 1],
            c='b'
        )
        ax.scatter(
            points[object_idxs][:, 0],
            points[object_idxs][:, 2],
            points[object_idxs][:, 1],
            c='r'
        )
        ax.scatter(
            points[receptacle_idxs][:, 0],
            points[receptacle_idxs][:, 2],
            points[receptacle_idxs][:, 1],
            c='g'
        )
        plt.show()

    print("Std agent: {}".format(np.std(points[agent_idxs])))
    print("Std o1: {}".format(np.std(points[object_idxs])))
    print("Std o2: {}".format(np.std(points[receptacle_idxs])))
    print("Std all: {}".format(np.std(points)))
    num_episodes = 0
    ep_ids = []
    ep_map = defaultdict(int)
    for i in range(len(agent_idxs)):
        idx = agent_idxs[i]
        for j in range(len(agent_idxs)):
            if i <= j:
                continue
            val1 = points[agent_idxs[i]]
            val2 = points[agent_idxs[j]]
            # val3 = points[object_idxs[i]]
            # val4 = points[object_idxs[j]]
            # val5 = points[receptacle_idxs[i]]
            # val6 = points[receptacle_idxs[j]]
            # val1 = points[object_idxs[i]]
            # val2 = points[receptacle_idxs[j]]
            dist = sim.geodesic_distance(val1, val2)
            # dist2 = sim.geodesic_distance(val3, val4)
            # dist3 = sim.geodesic_distance(val5, val6)
            if dist < 0.2: # and dist2 < 0.2: # or dist3 < 0.5:
                num_episodes += 1
                ep_map[i] += 1
                ep_map[j] += 1
                if ep_map[i] >= 5:
                    ep_ids.append(i)
                if ep_map[j] >= 5:
                    ep_ids.append(j)
        if i % 100 == 0:
            print("Num eps close to 0.5: {}/{} - {}".format(len(set(ep_ids)), (i+1), num_episodes))
    print("Num eps close to 0.5: {}/{} -- {}".format(len(set(ep_ids)), (i+1), num_episodes))

    ddddd = scipy.spatial.distance.pdist(points[agent_idxs], metric=sim.geodesic_distance)
    print("Min distance between points: {} - {} - {} -{}".format(np.min(ddddd), np.sum(ddddd < 0.5), ddddd.shape[0], len(agent_idxs)))
    ddddd = scipy.spatial.distance.pdist(points[object_idxs], metric=sim.geodesic_distance)
    print("Min distance between points: {} - {}".format(np.min(ddddd), np.sum(ddddd < 0.5)))
    ddddd = scipy.spatial.distance.pdist(points[receptacle_idxs], metric=sim.geodesic_distance)
    print("Min distance between points: {} - {}".format(np.min(ddddd), np.sum(ddddd < 0.5)))
    ddddd = scipy.spatial.distance.pdist(points, metric=sim.geodesic_distance)
    print("Min distance between points: {} - {}".format(np.min(ddddd), np.sum(ddddd < 0.5)))


def populate_episodes_points(episodes, scene_id):
    points = []
    for episode in episodes:
        if scene_id != episode["scene_id"]:
            continue

        point = str(episode["start_position"])
        points.append(episode["start_position"])
        if VISITED_POINT_DICT.get(point):
            VISITED_POINT_DICT[point] += 1
            # print("Redundant agent position in episode {}".format(episode["episode_id"]))
        else:
            VISITED_POINT_DICT[point] = 1

        for object_ in episode["objects"]:
            point = str(object_["position"])
            if VISITED_POINT_DICT.get(point):
                VISITED_POINT_DICT[point] += 1
                # print("Redundant point in episode {}".format(episode["episode_id"]))
            else:
                VISITED_POINT_DICT[point] = 1
            points.append(object_["position"])   
    return points


def is_valid_episode(sim, episode, near_dist, far_dist):
    agent_position = episode["start_position"]
    positions = [agent_position]
    objects = []
    for object_ in episode["objects"]:
        object_name = object_["objectHandle"].split("/")[-1].split(".")[0]
        position = object_["position"]
        positions.append(position)
        rotation = get_random_rotation()

        object_handle = get_object_handle(object_name)
        object_id  = sim.add_object_by_handle(object_handle)

        sim.set_rotation(quat_to_magnum(quat_from_coeffs(object_["rotation"])), object_id)
        sim.set_translation(mn.Vector3(position), object_id)

        tilt_threshold = 0.95
        orientation = sim.get_rotation(object_id)
        object_up = orientation.transform_vector(mn.Vector3(0,1,0))
        tilt = mn.math.dot(object_up, mn.Vector3(0,1,0))
        is_tilted = (tilt <= tilt_threshold)

        sim.remove_object(object_id)

        if is_tilted:
            print("\nEpsiode {}, tilted object: {}, contact: {}, rot coord: {}\n".format(episode["episode_id"], object_name, is_tilted, tilt))
            return False, episode

    episode["objects"] = objects

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i <= j:
                continue
            dist = get_geodesic_distance(sim, positions[i], positions[j])
            if not near_dist <= dist <= far_dist:
                return False, episode
    return True, episode


def get_random_rotation():
    angle = np.random.uniform(0, 2 * np.pi)
    rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    return rotation


def get_num_episodes_on_each_floor(points, floor_height_thresholds=[0.5, 1.0]):
    floor_count_map = defaultdict(int)
    y_points = [point[1] for point in points]
    y_points.sort()

    last_point = y_points[-1]
    print("Min point: {}".format(y_points[0]))
    print("Max point: {}".format(last_point))
    for point in y_points:
        dist = abs(point - last_point)
        if dist >= floor_height_thresholds[0]:
            floor_count_map[1] += 1
        elif dist >= floor_height_thresholds[1]:
            floor_count_map[2] += 1
        else:
            floor_count_map[0] += 1
    return floor_count_map


def get_all_tasks(path, scene_id, sim):
    tasks = []
    all_points = []
    for file_path in glob.glob(path + "/*.json"):
        with open(file_path, "r") as file:
            data = json.loads(file.read())
            if data["episodes"][0]["scene_id"] == scene_id:
                if ".json" in file_path:
                    tasks.append((data, file_path))
                    ep_points = populate_episodes_points(data["episodes"], scene_id)
                    floor_map = get_num_episodes_on_each_floor(ep_points)
                    all_points.extend(ep_points)
                    print("\nNum episodes per floor {}".format(floor_map))
    unique_points_count = len(VISITED_POINT_DICT.keys())
    print("Total tasks: {}".format(len(tasks)))
    print("Total unique points: {} -- {}".format(unique_points_count, unique_points_count / 3))
    return tasks, all_points


def get_sim(config):
    # Initialize simulator
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    return sim


def write_episode(dataset, filename):
    prefix = "data/tasks/" + filename
    with open(prefix, "w") as output_file:
        output_file.write(json.dumps(dataset))


def validate_tasks(
    config,
    d_lower_lim=5.0,
    d_upper_lim=30.0,
    prev_episodes="data/tasks",
    scene_id="empty_house.glb",
    show_plot=False,
    get_distance=True,
    check_tilt=False,
):
    sim = get_sim(config)
    # navMeshSettings = NavMeshSettings()
    # navMeshSettings.agent_max_climb = 0.5
    # sim.recompute_navmesh(sim.pathfinder, navMeshSettings, True)

    # Populate previously generated points
    tasks, all_points = get_all_tasks(prev_episodes, scene_id, sim)
    if get_distance:
        get_points_distance(np.array(all_points), sim, show_plot)
    sys.exit(1)

    results = []
    i = 0
    for task, file_path in tasks:
        episodes = task["episodes"]
        count = 0
        print(file_path)
        ep_ids = []
        for episode in episodes:
            if check_tilt:
                is_valid, ep_fixed = is_valid_episode(sim, episode, d_lower_lim, d_upper_lim)
            is_navigable = are_points_navigable(sim, episode)
            count += int(is_navigable)
            if not is_navigable:
                ep_ids.append(episode["episode_id"])

        i += 1

        print("\nScene: {}, Num valid episodes: {}, Total episodes: {}\n".format(scene_id, count, len(episodes)))
        print("Invalid episodes: {}".format(ep_ids))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate new episodes."
    )
    parser.add_argument(
        "--task-config",
        default="psiturk_dataset/rearrangement.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--scenes",
        help="Scenes",
        default="data/scene_datasets/habitat-test-scenes/empty_house.glb"
    )
    parser.add_argument(
        "--d_lower_lim",
        type=float,
        default=5,
        help="Closest distance between objects allowed.",
    )
    parser.add_argument(
        "--d_upper_lim",
        type=float,
        default=30.0,
        help="Farthest distance between objects allowed.",
    )
    parser.add_argument(
        "--prev_episodes",
        default="data/tasks",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--show-plot",
        dest='show_plot', action='store_true'
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
        validate_tasks(
            config,
            args.d_lower_lim,
            args.d_upper_lim,
            args.prev_episodes,
            scene_id,
            args.show_plot
        )
    else:
        print(f"Unknown dataset type: {dataset_type}")