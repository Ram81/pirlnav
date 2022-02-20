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


def get_all_tasks(path, scene_id):
    tasks = []
    for file_path in glob.glob(path + "/*.json"):
        with open(file_path, "r") as file:
            data = json.loads(file.read())
            if data["episodes"][0]["scene_id"] == scene_id:
                tasks.append((data, file_path))
                print(file_path)
    print("Total tasks: {}".format(len(tasks)))
    return tasks


def write_episode(dataset, filename):
    prefix = "data/tasks/" + filename
    with open(prefix, "w") as output_file:
        output_file.write(json.dumps(dataset))


def is_excluded_object_instruction(instruction, object_list):
    for object_name in object_list:
        if "the {}".format(object_name) in instruction:
            return True
    return False


def validate_tasks(
    config,
    d_lower_lim=5.0,
    d_upper_lim=30.0,
    prev_episodes="data/tasks",
    scene_id="empty_house.glb"
):
    # Populate previously generated points
    tasks = get_all_tasks(prev_episodes, scene_id)

    exclude_objects = config["TASK"]["EXCLUDE_OBJECTS"]
    object_name_map = dict(config["TASK"]["OBJECT_NAME_MAP"])

    exclude_objects = [object_name_map[object_handle] for object_handle in exclude_objects]
    print(exclude_objects)

    results = []
    i = 0
    src_task, src_file_path = tasks[0]
    dest_task, dest_file_path = tasks[1]

    src_episodes = src_task["episodes"]
    dest_episodes = dest_task["episodes"]
    
    count = 0
    file_name = dest_file_path.split("/")[-1].split(".")[0] + "_merged.json"
    merged_episodes = []
    ep_ids = []
    for episode_1, episode_2 in zip(src_episodes, dest_episodes):
        instruction = episode_1["task"]["instruction"]
        instruction_2 = episode_2["task"]["instruction"]
        if instruction != instruction_2:
            print("Instructions mismatch {}".format(episode_1["episode_id"]))
        if is_excluded_object_instruction(instruction, exclude_objects):
            print(instruction, episode_1["episode_id"], episode_2["episode_id"])
            merged_episodes.append(episode_2)
            ep_ids.append(episode_2["episode_id"])
            count += 1
        else:
            merged_episodes.append(episode_1)
            if episode_2["episode_id"] >= 400:
                ep_ids.append(episode_2["episode_id"])

    new_task = {
        "episodes": merged_episodes
    }

    write_episode(new_task, file_name)
    i += 1

    print("\nScene: {}, Num replaced episodes: {}, Total episodes: {}\n".format(scene_id, count, len(src_episodes)))
    print("\nReplaced episodes: {}".format(ep_ids))


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
            scene_id
        )
    else:
        print(f"Unknown dataset type: {dataset_type}")