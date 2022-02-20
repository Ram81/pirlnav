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
import math

import numpy as np
import magnum as mn
import matplotlib.pyplot as plt

from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_coeffs, quat_from_magnum, quat_to_coeffs
from mpl_toolkits.mplot3d import Axes3D


ISLAND_RADIUS_LIMIT = 1.5
VISITED_POINT_DICT = {}


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


def get_random_object_receptacle_pair(object_to_receptacle_map):
    object_to_receptacle_map_length = len(object_to_receptacle_map)
    index = np.random.choice(object_to_receptacle_map_length)
    print(object_to_receptacle_map[index])
    return object_to_receptacle_map[index]


def get_object_receptacle_list(object_to_receptacle_map):
    object_to_receptacle_list = []
    for object_, receptacles in object_to_receptacle_map.items():
        for receptacle in receptacles:
            object_to_receptacle_list.append((object_, receptacle))
    return object_to_receptacle_list


def get_object_receptacle_pair(object_to_receptacle_list, index):
    index = index % len(object_to_receptacle_list)
    return object_to_receptacle_list[index]


def use_in_for_receptacle(receptacle_name):
    for part in ["bowl", "cube", "bin", "basket"]:
        if part in receptacle_name.lower():
            return True
    return False


def get_task_config(config, object_name, receptacle_name, object_ids, receptacle_ids):
    task = {}
    in_or_on = "on"
    if use_in_for_receptacle(receptacle_name):
        in_or_on = "in"
    task["instruction"] = config["TASK"]["INSTRUCTION"].format(object_name, in_or_on, receptacle_name)
    task["type"] = config["TASK"]["TYPE"]
    task["goals"] = {}

    object_to_receptacle_map = {}
    for object_id, receptacle_id in zip(object_ids, receptacle_ids):
        if object_to_receptacle_map.get(object_id):
            object_to_receptacle_map[object_id].append(receptacle_id)
        else:
            object_to_receptacle_map[object_id] = [receptacle_id]

    task["goals"]["objectToReceptacleMap"] = object_to_receptacle_map
    return task


def build_episode(config, episode_id, objects, agent_position, agent_rotation, object_name, receptacle_name):
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
    
    episode["task"] = get_task_config(config, object_name, receptacle_name, object_ids, receptacle_ids)
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
    ylim=None, zlim=None, object_names=[], is_tilted_or_colliding=[]
):
    bad_points = np.zeros(points.shape[0], dtype=bool)
    # Outside X, Y, or Z limits
    if xlim:
        bad_points[points[:, 0] < xlim[0]] = 1
        bad_points[points[:, 0] > xlim[1]] = 1

    if ylim:
        bad_points[points[:, 1] < ylim[0]] = 1
        bad_points[points[:, 1] > ylim[1]] = 1

    if zlim:
        bad_points[points[:, 2] < zlim[0]] = 1
        bad_points[points[:, 2] > zlim[1]] = 1

    for i, point in enumerate(points):
        point_list = point.tolist()
        existing_point_count = VISITED_POINT_DICT.get(str(point_list))
        is_navigable = sim.is_navigable(point)
        if existing_point_count is not None and existing_point_count >= 1 or is_tilted_or_colliding[i] or not is_navigable:
            bad_points[i] = 1

    # Too close to another object or receptacle
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i == j:
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
                bad_points[i] = 1

    return bad_points


def rejection_sampling(
    sim, points, rotations, d_lower_lim, d_upper_lim,
    geodesic_to_euclid_min_ratio, xlim=None,
    ylim=None, zlim=None, num_tries=10000, object_names=[], scene_bb=None,
    is_tilted_or_colliding=[]
):
    bad_points = get_bad_points(
        sim, points, rotations, d_lower_lim, d_upper_lim,
        geodesic_to_euclid_min_ratio, xlim, ylim,
        zlim, object_names, is_tilted_or_colliding
    )

    while sum(bad_points) > 0 and num_tries > 0:

        for i, bad_point in enumerate(bad_points):
            if bad_point and i == 0:
                points[i] = get_random_point(sim)
            elif bad_point:
                points[i], rotations[i], is_tilted_or_colliding[i] = get_random_object_position(sim, object_names[i - 1])

        bad_points = get_bad_points(
            sim, points, rotations, d_lower_lim, d_upper_lim,
            geodesic_to_euclid_min_ratio, xlim, ylim,
            zlim, object_names, is_tilted_or_colliding
        )
        num_tries -= 1
    
    print(sum(bad_points), num_tries)

    if sum(bad_points) > 0:
        print("\n Error generating unique points, try using bigger retries")
        # sys.exit(1)

    return points, bad_points


def get_random_point(sim):
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


def get_random_object_position(sim, object_name, scene_bb=None, scene_collision_margin=0.04, tilt_threshold=0.95):
    object_handle = get_object_handle(object_name)
    position = get_random_point(sim)

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

    step_physics_n_times(sim)

    translation = sim.get_translation(object_id)
    rotation = sim.get_rotation(object_id)

    orientation = sim.get_rotation(object_id)
    object_up = orientation.transform_vector(mn.Vector3(0,1,0))
    tilt = mn.math.dot(object_up, mn.Vector3(0,1,0))
    is_tilted = (tilt <= tilt_threshold)

    adjusted_translation = mn.Vector3(
        0, scene_collision_margin, 0
    ) + translation
    sim.set_translation(adjusted_translation, object_id)
    is_colliding = sim.contact_test(object_id)

    is_tilted_or_colliding = (is_tilted or is_colliding)

    rotation = quat_to_coeffs(quat_from_magnum(rotation)).tolist()
    translation = np.array(translation).tolist()
    sim.remove_object(object_id)
    return np.array(translation), rotation, is_tilted_or_colliding


def populate_episodes_points(episodes, scene_id):
    for episode in episodes["episodes"]:
        if scene_id != episode["scene_id"]:
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


def populate_prev_generated_points(path, scene_id):
    for file_path in glob.glob(path + "/*.json"):
        with open(file_path, "r") as file:
            data = json.loads(file.read())
            populate_episodes_points(data, scene_id)
    print("Total previously generated points {}".format(len(VISITED_POINT_DICT.keys())))


def remove_all_objects(sim):
    for object_id in sim.get_existing_object_ids():
        sim.remove_object(object_id)
    sim.clear_recycled_object_ids()


def generate_points(
    config,
    objs_per_rec,
    num_episodes,
    num_targets,
    number_retries_per_target=1000,
    d_lower_lim=5.0,
    d_upper_lim=30.0,
    geodesic_to_euclid_min_ratio=1.1,
    prev_episodes="data/tasks",
    scene_id="empty_house.glb",
    use_google_objects=False,
    output_path="data/tasks/big_house.json",
):
    # Initialize simulator
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    # Populate previously generated points
    populate_prev_generated_points(prev_episodes, scene_id)

    episode_count = 0
    episodes = []
    
    object_to_receptacle_list = get_object_receptacle_list(config["TASK"]["OBJECTS_RECEPTACLE_MAP"])

    google_object_to_receptacle_list = get_object_receptacle_list(config["TASK"]["GOOGLE_OBJECT_RECEPTACLE_MAP"])
    if use_google_objects:
        object_to_receptacle_list.extend(google_object_to_receptacle_list)
        # object_to_receptacle_list = google_object_to_receptacle_list

    object_name_map = dict(config["TASK"]["OBJECT_NAME_MAP"])
    y_limit = config["TASK"].get("Y_LIMIT")
    x_limit = None
    if config["TASK"].get("X_LIMIT"):
        x_limit = config["TASK"]["X_LIMIT"]
    num_points = config["TASK"]["NUM_OBJECTS"] + config["TASK"]["NUM_RECEPTACLES"] + 1

    all_points = []
    num_episodes = num_episodes * len(object_to_receptacle_list)
    print("Generating total {} episodes for {} object receptacle pair".format(num_episodes, len(object_to_receptacle_list)))

    print("\n\n\n YCB object receptacle pairs: {}".format(len(object_to_receptacle_list)))
    print("\n\n\n Google object receptacle pairs: {}".format(len(google_object_to_receptacle_list)))
    while episode_count < num_episodes or num_episodes < 0:
        for object_list in object_to_receptacle_list:
            print("\nEpisode {}\n".format(episode_count))
            object_, receptacle = object_list
            objects = []

            object_name = object_name_map[object_]
            receptacle_name = object_name_map[receptacle]

            points = []
            rotations = []
            is_tilted_or_colliding = []
            for idx in range(num_points):
                is_invalid = False
                if idx == 0:
                    point = get_random_point(sim)
                    rotation = get_random_rotation()
                else:
                    point, rotation, is_invalid = get_random_object_position(sim, object_list[idx - 1])
                points.append(point)
                rotations.append(rotation)
                is_tilted_or_colliding.append(is_invalid)
            
            points = np.array(points)
            points, bad_points = rejection_sampling(
                sim, points, rotations, d_lower_lim, d_upper_lim,
                geodesic_to_euclid_min_ratio, xlim=x_limit, ylim=y_limit,
                num_tries=number_retries_per_target, object_names=object_list,
                is_tilted_or_colliding=is_tilted_or_colliding
            )

            if sum(bad_points) > 0:
                continue

            # Mark valid points as visited to get unique points
            print("Total unique points: {}".format(len(VISITED_POINT_DICT.keys())))
            for point in points:
                VISITED_POINT_DICT[str(point.tolist())] = 1
                all_points.append(point.tolist())
                
            agent_position = points[0].tolist()
            agent_rotation = rotations[0]

            source_position = points[1].tolist()
            source_rotation = rotations[1]

            target_position = points[2].tolist()
            target_rotation = rotations[2]

            # Create episode object configs
            objects.append(build_object(object_, len(objects), object_name, False, source_position, source_rotation))
            objects.append(build_object(receptacle, len(objects), receptacle_name, True, target_position, target_rotation))
            
            # Build episode from object and agent initilization.
            episode = build_episode(config, episode_count, objects, agent_position,
                agent_rotation, object_name, receptacle_name)
            episodes.append(episode)

            remove_all_objects(sim)
            episode_count += 1

            if episode_count % 10 == 0:
                dataset = {
                    "episodes": episodes
                }
                write_episode(dataset, output_path)

    dataset = {
        "episodes": episodes
    }
    return dataset


def write_episode(dataset, filename):
    #prefix = "data/tasks/" + filename
    with open(filename, "w") as output_file:
        output_file.write(json.dumps(dataset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate new episodes."
    )
    parser.add_argument(
        "--task-config",
        default="psiturk_dataset/task/rearrangement.yaml",
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
        help="Number of target points to sample",
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
        default=5.0,
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
        default=1,
        help="Number of objects per goal.",
    )
    parser.add_argument(
        "--prev_episodes",
        default="data/tasks",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--use_google_objects",
        dest='use_google_objects', action='store_true',
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
            args.use_google_objects,
            args.output
        )
        write_episode(dataset, args.output)
    else:
        print(f"Unknown dataset type: {dataset_type}")