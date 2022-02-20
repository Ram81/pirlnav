#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import glob
import gzip
import itertools
import json
import lzma
import multiprocessing
import os
import os.path as osp
import random
import sys
import traceback
from collections import defaultdict

import GPUtil
import numpy as np
import pydash
import tqdm

import habitat
import habitat_sim
from habitat.config.default import get_config
from habitat.core.dataset import ObjectInScene, SceneState
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.core.simulator import AgentState
from habitat.datasets import make_dataset

# from habitat.datasets.object_nav import object_nav_dataset
from habitat.datasets.object_nav.create.object_nav_generator import (
    generate_objectnav_episode,
)

# from habitat.datasets.object_nav.create.object_nav_generator_old import (
#     generate_new_obj_nav_episode,
# )
from habitat.datasets.pointnav.create.utils import (
    get_gibson_scenes,
    get_habitat_gibson_scenes,
    get_mp3d_scenes,
)

CFG_TEST = "configs/test/habitat_mp3d_eqa_test.yaml"
CLOSE_STEP_THRESHOLD = 0.028

COMPRESSION = ".gz"
OUTPUT_OBJ_FOLDER = (
    "./data/datasets/objectnav/mp3d/v1_allobjs_quaternion_fixed_2"
)

scenes_source = "mp3d"
OUTPUT_JSON_FOLDER = (
    f"./data/datasets/objectnav/{scenes_source}/objnav_ycb_objs_200k_single/"
)
# "./data/datasets/objectnav/mp3d/objnav_challenge2020_final_q_fixed_ud_50k_train_split_small_1m_local_planner_min/"
# "./data/datasets/objectnav/mp3d/v1_floatencoded_cats_90_normalized_round_obb_fixed_6"
NUM_GPUS = len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 12
tiny_gibson_scenes = [
    "Allensville",
    "Beechwood",
    "Benevolence",
    "Coffeen",
    "Collierville",
    "Corozal",
    "Cosmos",
    "Darden",
    "Forkland",
    "Hanson",
    "Hiteman",
    "Ihlen",
    "Klickitat",
    "Lakeville",
    "Leonardo",
    "Lindenwood",
    "Markleeville",
    "Marstons",
    "McDade",
    "Merom",
    "Mifflinburg",
    "Muleshoe",
    "Newfields",
    "Noxapater",
    "Onaga",
    "Pinesdale",
    "Pomaria",
    "Ranchester",
    "Shelbyville",
    "Stockman",
    "Tolstoy",
    "Uvalda",
    "Wainscott",
    "Wiconisco",
    "Woodbine",
]

LARGE_MP3D_SCENE = "1pXnuDYAj8r"

mp3dcat40_map = defaultdict(lambda: 41)
mp3dcat40_mapinv = dict()
mp3dcat_nametosynset = dict()
with open("examples/mpcat40.tsv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for line in reader:
        id_ = int(line[0].strip())
        synsets = line[3].strip().split(",")
        if len(synsets) == 0:
            continue
        mp3dcat40_mapinv[id_] = line[1].strip()
        for synset in synsets:
            mp3dcat40_map[synset] = id_
        mp3dcat_nametosynset[line[1].strip()] = synset

# print(mp3dcat_nametosynset)


def read_ycb_google_objects(
    path="habitat/datasets/object_nav/create/ycb_google_16k_meta.csv",
    ycb_objects_dir="data/assets/objects/ycb_google_16k_v2/configs_gltf/",
):
    with open(path, "r") as f:
        csv_r = csv.reader(f)
        headers = next(csv_r)
        filename_ind = headers.index("filename")
        object_name_ind = headers.index("object_name")
        synset_ind = headers.index("synset")
        object_category_ind = headers.index("category")
        ycb_objects_lib = [
            {
                "filename": ycb_objects_dir + line[filename_ind],
                "object_name": line[object_name_ind],
                "synset": line[synset_ind],
                "object_category": line[object_category_ind],
            }
            for line in csv_r
        ]
        ycb_objects_categories = {
            object["object_category"] for object in ycb_objects_lib
        }
        return ycb_objects_lib, sorted(list(ycb_objects_categories))


ycb_objects_lib, ycb_objects_categories = read_ycb_google_objects()
ycb_objects_categories = [
    "foodstuff",
    "stationery",
    "fruit",
    "plaything",
    "hand_tool",
    "game_equipment",
    "kitchenware",
]


def sample_objects_for_episode(
    objects_lib, num_uniq_selected_objects=5, num_copies=3
):
    # Up to 5 unique objects in the scene
    uniq_selected_objects = np.random.choice(
        objects_lib,
        size=np.random.randint(low=1, high=num_uniq_selected_objects),
    )
    if num_copies > 1:
        # Up to 2 instances of the same object
        selected_objects = [
            object.copy()
            for object in uniq_selected_objects
            for _ in range(np.random.randint(low=1, high=num_copies))
        ]
        return selected_objects
    return uniq_selected_objects


def sample_scene_state(selected_objects):
    max_semantic_object_id = 1 << 16  #  shift to avoid using first 2 bytes
    # If scene is loaded and sim is available.
    # sem_scene = sim.semantic_annotations()
    # if sem_scene and len(sem_scene.objects) > 0:
    #     max_semantic_object_id = max(
    #         [int(obj.id.split("_")[-1]) for obj in sem_scene.objects])

    for object_id, object in enumerate(selected_objects):
        object["object_id"] = str(max_semantic_object_id + object_id)

    return SceneState(
        objects=[
            ObjectInScene(
                object_id=str(max_semantic_object_id + object_id),
                object_template=f"{object['filename'].replace('.glb', '')}",
                scale=5,
                semantic_category_id=category_to_mp3d_category_id[
                    object["object_category"]
                ],
            )
            for object_id, object in enumerate(selected_objects)
        ]
    )


# with open("examples/coco_to_synset_edited.json", "r") as f:
#     coco_map = json.load(f)

mp3d_valid_ids = np.array(
    [3]
    + list(range(5, 9))
    + list(range(10, 12))
    + list(range(13, 16))
    + list(range(18, 21))
    + list(range(22, 24))
    + list(range(25, 28))
    + list(range(33, 35))
    + [38]
)

# np.array(list(set(mp3dcat40_map[synset] for synset in coco_map.values())))
mp3d_valid_ids = np.sort(mp3d_valid_ids)
mp3d_wordlist = [mp3dcat40_mapinv[a] for a in mp3d_valid_ids]
category_to_task_category_id = {k: int(v) for v, k in enumerate(mp3d_wordlist)}
category_to_mp3d_category_id = {
    k: int(v) for k, v in zip(mp3d_wordlist, mp3d_valid_ids)
}

# Appending YCB categories
max_task_cat = max(category_to_task_category_id.values())
max_mp3d_cat = 42  # Max MP3D cat value
for ycb_category_id, category in enumerate(ycb_objects_categories):
    category_to_task_category_id[category] = max_task_cat + ycb_category_id + 1
    category_to_mp3d_category_id[category] = max_mp3d_cat + ycb_category_id + 1

# print(ycb_objects_categories)
# print(category_to_task_category_id)


def get_objnav_config(i, scene):

    CFG_EQA = "configs/tasks/challenge_objectnav2021.local.rgbd.yaml"
    objnav_config = get_config(CFG_EQA).clone()
    objnav_config.defrost()
    # print(objnav_config.SIMULATOR.AGENT_0.RADIUS)
    # objnav_config.SIMULATOR_GPU_ID = i % 2
    objnav_config.TASK.SENSORS = []
    objnav_config.SIMULATOR.AGENT_0.SENSORS = ["SEMANTIC_SENSOR"]
    # objnav_config.SIMULATOR.AGENT_0.RADIUS = 0.2
    # HEIGHT = objnav_config.SIMULATOR.AGENT_0.HEIGHT
    # objnav_config.SIMULATOR.AGENT_0.HEIGHT = HEIGHT
    # objnav_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 64
    # objnav_config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 64
    # objnav_config.SIMULATOR.RGB_SENSOR.HEIGHT = 64
    # objnav_config.SIMULATOR.RGB_SENSOR.WIDTH = 64
    FOV = 90
    objnav_config.SIMULATOR.RGB_SENSOR.HFOV = FOV
    # objnav_config.SIMULATOR.RGB_SENSOR.VFOV = FOV
    # objnav_config.SIMULATOR.RGB_SENSOR.POSITION = [0, HEIGHT, 0]
    objnav_config.SIMULATOR.DEPTH_SENSOR.HFOV = FOV
    # objnav_config.SIMULATOR.DEPTH_SENSOR.VFOV = FOV
    # objnav_config.SIMULATOR.DEPTH_SENSOR.POSITION = [0, HEIGHT, 0]
    objnav_config.SIMULATOR.SEMANTIC_SENSOR.HFOV = FOV
    # objnav_config.SIMULATOR.SEMANTIC_SENSOR.VFOV = FOV
    # objnav_config.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0, HEIGHT, 0]

    # TODO lower resolution of semantic sensor
    objnav_config.TASK.MEASUREMENTS = []
    # objnav_config.TASK.POSSIBLE_ACTIONS = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']
    deviceIds = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
    )
    if i < NUM_GPUS * TASKS_PER_GPU or len(deviceIds) == 0:
        deviceId = i % NUM_GPUS
    else:
        deviceId = deviceIds[0]
    objnav_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
        deviceId  # i % NUM_GPUS
    )
    objnav_config.DATASET.DATA_PATH = (
        "./data/datasets/pointnav/mp3d/v1/val/val.json.gz"
    )
    objnav_config.DATASET.SCENES_DIR = "./data/scene_datasets/"
    objnav_config.DATASET.SPLIT = "train"
    # objnav_config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    # objnav_config.TASK.SENSORS.append("HEADING_SENSOR")
    # new_test_scene = "/private/home/maksymets/data/gibson/gibson_tiny_manual_verified/" + tiny_gibson_scenes[0] + '.glb'
    # new_test_scene = scene#'data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb'
    objnav_config.SIMULATOR.SCENE = scene  # new_test_scene
    objnav_config.freeze()
    return objnav_config


def get_simulator(objnav_config):
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.SIMULATOR)
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = objnav_config.SIMULATOR.AGENT_0.RADIUS
    navmesh_settings.agent_height = objnav_config.SIMULATOR.AGENT_0.HEIGHT
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


def get_gravity_mobb(object_obb: habitat_sim.geo.OBB):
    bounding_area = [
        (object_obb.local_to_world @ np.array([x, y, z, 1]))[:-1]
        for x, y, z in itertools.product(*([[-1, 1]] * 3))
    ]
    bounding_area = np.array(bounding_area, dtype=np.float32)
    # print('Bounding Area: %s' % bounding_area)
    # TODO Maybe Cache this
    return habitat_sim.geo.compute_gravity_aligned_MOBB(
        habitat_sim.geo.GRAVITY, bounding_area
    )


def get_goals_view_points(sim, goal_objects, agent_state):
    from habitat.tasks.nav.object_nav_task import (
        ObjectGoal,
        ObjectGoalNavEpisode,
        ObjectViewLocation,
    )

    goals = [
        ObjectGoal(
            object_id=object["object_id"],
            object_category=object["object_category"],
            position=object["position"],
            view_points=[
                ObjectViewLocation(
                    agent_state=AgentState(
                        position=np.array(
                            sim.pathfinder.snap_point(object["position"])
                        )
                    ),
                    iou=0.5 # TODO add some legit value
                )
            ],
        )
        for object in goal_objects
    ]

    goal_targets = (
        [vp.agent_state.position for vp in goal.view_points] for goal in goals
    )

    closest_goal_targets = (
        sim.geodesic_distance(agent_state, vps) for vps in goal_targets
    )
    # closest_goal_targets, goals_sorted = zip(
    #     *sorted(
    #         zip(closest_goal_targets, goals),
    #         key=lambda x: x[0],
    #     )
    # )
    return goals, list(closest_goal_targets)


def reset_goal_objects(sim, goal_objects, agent_state, closest_goal_targets):
    retries = 0
    from habitat_sim.physics import MotionType

    while np.inf in closest_goal_targets and retries < 20:
        retries += 1

        for object_order, object in enumerate(goal_objects):
            if closest_goal_targets[object_order] != np.inf and retries < 10:
                continue
            sim_object_id = sim.objid_to_sim_object_mapping[
                object["object_id"]
            ]
            sim.set_object_motion_type(MotionType.KINEMATIC, sim_object_id)
            # sim.set_translation(sim.pathfinder.get_random_navigable_point(), sim_object_id), sim.get_translation(sim_object_id)
            while not sim.sample_object_state(sim_object_id):
                pass
            sim.set_object_motion_type(MotionType.STATIC, sim_object_id)
            object["position"] = list(sim.get_translation(sim_object_id))
            object["rotation"] = list(sim.get_rotation_vec(sim_object_id))
            object["obb"] = sim.get_object_scene_node(
                sim_object_id
            ).cumulative_bb
            sim.recompute_navmesh(sim.pathfinder, sim.navmesh_settings, True)
        if np.all(closest_goal_targets == np.inf):
            s = sim.sample_navigable_point()
        goals, closest_goal_targets = get_goals_view_points(
            sim, goal_objects=goal_objects, agent_state=agent_state
        )
        # logger.info(f"closest_goal_targets: {closest_goal_targets} rt {retries}")


def fill_episode(
    episode,
    agent_state,
    episode_counter,
    goal_category,
    goals,
    selected_objects,
    sim,
):
    episode.start_position = agent_state
    angle = np.random.uniform(0, 2 * np.pi)
    episode.start_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    sim.set_agent_state(agent_state, episode.start_rotation)
    episode.goals = goals
    episode_counter += 1
    episode.episode_id = str(episode_counter)
    episode.object_category = goal_category
    from habitat.core.dataset import ObjectInScene, SceneState

    episode.scene_state = SceneState(
        objects=[
            ObjectInScene(
                object_id=object["object_id"],
                object_template=f"{object['filename'].replace('.glb', '')}",
                rotation=object["rotation"],
                position=object["position"],
                scale=5,
                semantic_category_id=category_to_mp3d_category_id[
                    object["object_category"]
                ],
            )
            for object in selected_objects
        ]
    )


def reset_episode(
    episode, sim, selected_objects, near_dist=1, far_dist=5, episode_counter=0
):

    for object in selected_objects:
        sim_object_id = sim.objid_to_sim_object_mapping[object["object_id"]]
        object["position"] = list(sim.get_translation(sim_object_id))
        object["rotation"] = list(sim.get_rotation_vec(sim_object_id))
        object["obb"] = sim.get_object_scene_node(sim_object_id).cumulative_bb

    category_to_object = {}
    for object in selected_objects:
        category_to_object[object["object_category"]] = category_to_object.get(
            object["object_category"], []
        ) + [object]
    goal_category = np.random.choice(list(category_to_object.keys()))

    from habitat.datasets.pointnav.pointnav_generator import (
        ISLAND_RADIUS_LIMIT,
    )

    s = sim.sample_navigable_point()
    while sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        s = sim.sample_navigable_point()

    goals, closest_goal_targets = get_goals_view_points(
        sim, goal_objects=category_to_object[goal_category], agent_state=s
    )

    reset_goal_objects(
        sim,
        goal_objects=category_to_object[goal_category],
        agent_state=s,
        closest_goal_targets=closest_goal_targets,
    )

    retries = 0
    while (
        np.inf in closest_goal_targets
        or not near_dist <= min(closest_goal_targets) <= far_dist
    ):
        retries += 1
        if retries > 200 and np.inf in closest_goal_targets:
            logger.info(f"Falling back to previous episode as dist is inf.")
            return None
        s = sim.sample_navigable_point()
        goals, closest_goal_targets = get_goals_view_points(
            sim, goal_objects=category_to_object[goal_category], agent_state=s
        )

    fill_episode(
        episode,
        s,
        episode_counter,
        goal_category,
        goals,
        selected_objects,
        sim,
    )

    return episode


def generate_scene(args):
    i, scene, split = args
    objnav_config = get_objnav_config(i, scene)

    sim = get_simulator(objnav_config)
    # Check there exists a navigable point
    test_point = sim.sample_navigable_point()
    if not sim.is_navigable(np.array(test_point)):
        print("Scene is not navigable: %s" % scene)
        sim.close()
        return scene, 0, defaultdict(list), None

    # goals_by_class = defaultdict(list)
    # cell_size = objnav_config.SIMULATOR.AGENT_0.RADIUS / 2.0
    # for obj in tqdm.tqdm(selected_objects, desc="Objects for %s:" % scene):
    #     print("Object id: %d" % obj["id"])
    #     print(obj["category_name"])
    #
    #     goal = build_goal(
    #         sim,
    #         object_id=obj["object_id"],
    #         object_name_id=obj["object_name"],
    #         object_category_name=obj["category_name"],
    #         object_category_id=1, #obj["category_id"],
    #         object_position=obj["position"],
    #         object_aabb=obj["aabb"],
    #         object_obb=obj["obb"],
    #         object_gmobb=obj["gravity_mobb"],
    #         cell_size=cell_size,
    #         grid_radius=3.0,
    #     )
    #     if goal == None:  # or len(goal.view_points) == 0:
    #         continue
    #     goals_by_class[obj["category_id"]].append(goal)
    #     # break
    # os.makedirs(osp.dirname(fname_obj), exist_ok=True)
    # total_objects_by_cat = {k: len(v) for k, v in goals_by_class.items()}
    # with open(fname_obj, "wb") as f:
    #     pickle.dump(goals_by_class, f)

    scene_key = habitat.Dataset.scene_from_scene_path(scene)
    fname = (
        f"{OUTPUT_JSON_FOLDER}/{split}/content/{scene_key}.json{COMPRESSION}"
    )

    if os.path.exists(fname):
        print("Scene already generated. Skipping")
        sim.close()
        return scene, 0, 0, None

    total_episodes = 2000 if split == "train" else 10
    # if split == 'val':
    #    total_obs_json = 100
    # if split == 'val_mini':
    #    total_episodes = 30
    dset = habitat.datasets.make_dataset("ObjectNav-v1")
    dset.category_to_task_category_id = category_to_task_category_id
    dset.category_to_mp3d_category_id = category_to_mp3d_category_id
    eps_generated = 0
    with tqdm.tqdm(total=total_episodes, desc=scene) as pbar:

        try:
            while eps_generated < total_episodes:
                sim.habitat_config.defrost()
                selected_objects = sample_objects_for_episode(ycb_objects_lib)
                sim.habitat_config.scene_state = [
                    sample_scene_state(selected_objects).__dict__
                ]
                sim.habitat_config.freeze()
                sim._initialize_scene()

                for object in selected_objects:
                    sim_object_id = sim.objid_to_sim_object_mapping[
                        object["object_id"]
                    ]
                    object["position"] = list(
                        sim.get_translation(sim_object_id)
                    )
                    object["rotation"] = list(
                        sim.get_rotation_vec(sim_object_id)
                    )
                    object["obb"] = sim.get_object_scene_node(
                        sim_object_id
                    ).cumulative_bb

                category_to_object = {}
                for object in selected_objects:
                    category_to_object[
                        object["object_category"]
                    ] = category_to_object.get(
                        object["object_category"], []
                    ) + [
                        object
                    ]

                from habitat.core.simulator import AgentState
                from habitat.tasks.nav.object_nav_task import (
                    ObjectGoal,
                    ObjectGoalNavEpisode,
                    ObjectViewLocation,
                )

                # all_view_points = [np.array(
                #                             sim.pathfinder.snap_point(
                #                                 object['position']))
                #             for object in selected_objects]
                #
                # source_position = sim.sample_navigable_point()
                # while sim.geodesic_distance(source_position, all_view_points) is not math.inf:
                #     source_position = sim.sample_navigable_point()
                # angle = np.random.uniform(0, 2 * np.pi)
                # source_rotation = [
                #     0,
                #     np.sin(angle / 2),
                #     0,
                #     np.cos(angle / 2),
                # ]  # Pick random starting rotation

                for goal_category in category_to_object.keys():
                    # Generate episode for each category
                    # goal_category = \
                    # np.random.choice(list(category_to_object.keys()), size=1)[
                    #     0]
                    try:
                        for ep in generate_objectnav_episode(
                            sim,
                            goals=[
                                ObjectGoal(
                                    object_id=object["object_id"],
                                    object_category=object["object_category"],
                                    position=object["position"],
                                    view_points=[
                                        ObjectViewLocation(
                                            agent_state=AgentState(
                                                position=np.array(
                                                    sim.pathfinder.snap_point(
                                                        object["position"]
                                                    )
                                                )
                                            ),
                                        )
                                    ],
                                )
                                for object in category_to_object[goal_category]
                            ],
                            scene_state=SceneState(
                                objects=[
                                    ObjectInScene(
                                        object_id=object["object_id"],
                                        object_template=f"{object['filename'].replace('.glb', '')}",
                                        rotation=object["rotation"],
                                        position=object["position"],
                                        scale=5,
                                        semantic_category_id=category_to_mp3d_category_id[
                                            object["object_category"]
                                        ],
                                    )
                                    for object in selected_objects
                                ]
                            ),
                            num_episodes=1,
                        ):
                            ep.episode_id = eps_generated
                            dset.episodes.append(ep)
                            pbar.update()
                            eps_generated += 1
                    except RuntimeError:
                        pbar.update()

                    # episode = ObjectGoalNavEpisode(
                    #     episode_id=str(eps_generated),
                    #     scene_id=sim.habitat_config.SCENE,
                    #     scene_state=SceneState(
                    #         objects=[ObjectInScene(
                    #             object_id=str(object_id),
                    #             object_template=f"{object['filename'].replace('.glb', '')}",
                    #             rotation=object['rotation'],
                    #             position=object['position'],
                    #             scale=5,
                    #             semantic_id=category_to_mp3d_category_id[
                    #                 object["object_category"]]
                    #         ) for object_id, object in
                    #             enumerate(selected_objects)]
                    #     ),
                    #     start_position=source_position,
                    #     start_rotation=source_rotation,
                    #     object_category=goal_category,
                    #     goals=[
                    #         ObjectGoal(
                    #             object_id=str(object['object_id']),
                    #             object_category=object['object_category'],
                    #             position=object['position'],
                    #             view_points=[ObjectViewLocation(
                    #                 agent_state=AgentState(
                    #                     position=np.array(
                    #                         sim.pathfinder.snap_point(
                    #                             object['position']))
                    #                 )
                    #             )]
                    #         )
                    #         for object in category_to_object[goal_category]
                    #     ],
                    # )
                # print("\n +1 iterations")
        except RuntimeError:
            traceback.print_exc()
            print("Skipping category")

    for ep in dset.episodes:
        # STRIP OUT PATH
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    os.makedirs(osp.dirname(fname), exist_ok=True)
    save_dataset(dset, fname)
    print(f"Saved dataset {fname}.")
    # dset2 = habitat.datasets.make_dataset('ObjectNav-v1')
    # with gzip.open(fname, 'rt') as f:
    #     #dset2f.read()
    #     dset2.from_json(f.read())
    # assert dset2.category_vocab_dict_dense.word_list == dset.category_vocab_dict_dense.word_list
    # assert dset2.category_vocab_dict_sparse == dset.category_vocab_dict_sparse
    sim.close()
    return scene, eps_generated, 0, fname


SPLIT = None


def get_file_opener(fname):
    ext = os.path.splitext(fname)[-1]

    if ext == ".gz":
        file_opener = gzip.open
    elif ext == ".xz":
        file_opener = lzma.open
    else:
        print(ext)
        assert False
        # ile_opener = open
    return file_opener


def save_dataset(dset: habitat.Dataset, fname: str):
    file_opener = get_file_opener(fname)
    # compression = gzip if format == 'gzip' else lzma
    # dset = dset.from_json(dset.to_json())
    # ddset = dset.from_json(json.loads(json.dumps(dset.to_json())))
    if (
        os.path.basename(os.path.dirname(fname)) == "content"
        and len(dset.episodes) == 0
    ):
        print("WARNING UNEXPECTED EMPTY EPISODES: %s" % fname)
        return
    with file_opener(fname, "wt") as f:
        if len(dset.episodes) == 0:
            print("WARNING EMPTY EPISODES: %s" % fname)
            f.write(
                json.dumps(
                    {
                        "episodes": [],
                        "category_to_task_category_id": dset.category_to_task_category_id,
                        "category_to_mp3d_category_id": dset.category_to_mp3d_category_id,
                    }
                )
            )
        else:
            # dset = object_nav_dataset.ObjectNavDatasetV1.dedup_goals_dset(dset)
            f.write(dset.to_json())


def read_dset(json_fname):
    print(json_fname)
    dset2 = habitat.datasets.make_dataset("ObjectNav-v1")
    file_opener = get_file_opener(json_fname)

    # compression = gzip if os.path.splitext(json_fname)[-1] == 'gz' else lzma
    with file_opener(json_fname, "rt") as f:
        dset2.from_json(f.read())
    return dset2


if __name__ == "__main__":
    mp_ctx = multiprocessing.get_context("fork")

    new_test_scene = (
        "/private/home/maksymets/data/gibson/gibson_tiny_manual_verified/"
        + tiny_gibson_scenes[0]
        + ".glb"
    )

    a = dict()
    split = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "run"
    # read_dset('./data/datasets/objectnav/mp3d/objnav_challenge2020_v2/val/content/2azQ1b91cZZ.json.gz')
    # Just for testing
    def prepare_args(split):
        print(split)
        if split in ["test_standard", "test_challenge"]:
            filename = "examples/mp3d_" + split + ".txt"
            print(filename)
            with open(filename, "r") as f:
                # print(f.read())
                scenes = [line.strip() for line in f]
                print(scenes)
        else:
            if scenes_source == "gibson":
                scenes = get_gibson_scenes(split)  #  get_gibson_scenes(split)
            else:
                scenes = get_mp3d_scenes(split)
                # scenes = [LARGE_MP3D_SCENE]  # debug remove

        random.shuffle(scenes)
        if scenes_source == "gibson":
            scenes = [
                f"./data/scene_datasets/gibson/{scene}.glb" for scene in scenes
            ]

        else:
            scenes = [
                f"./data/scene_datasets/mp3d/{scene}/{scene}.glb"
                for scene in scenes
            ]

        blacklist = ["JmbYfDe2QKZ"]
        # if split != 'val' assert
        scenes = [
            scene
            for scene in scenes
            if os.path.basename(os.path.dirname(scene)) not in blacklist
        ]
        if split == "val_mini":
            scenes = [scenes[0]]
        return [(i, scene, split) for i, scene in enumerate(scenes)]

    np.random.seed(1234)
    if split == "*":
        args = []
        for split in ["train", "test", "val"]:
            args += prepare_args(split)
    else:
        args = prepare_args(split)

    ############### GENERATE EPISODES ########################
    GPU_THREADS = NUM_GPUS * TASKS_PER_GPU  # 1  #debug
    print(GPU_THREADS)
    print("*" * 1000)
    CPU_THREADS = int(multiprocessing.cpu_count() / 4) if mode == "run" else 1
    print(f"CPU_THREADS: {CPU_THREADS}")
    with mp_ctx.Pool(CPU_THREADS, maxtasksperchild=1) as pool, tqdm.tqdm(
        total=len(args)
    ) as pbar, open("train_subtotals.json", "w") as f:
        total_all = 0
        subtotals = []
        for scene, subtotal, subtotal_by_cat, fname in pool.imap_unordered(
            generate_scene, args
        ):
            a[scene] = (subtotal, subtotal_by_cat)
            # print("*" * 10)
            pbar.update()
            # print(a)
            total_all += subtotal
            subtotals.append(subtotal_by_cat)
        print(total_all)
        print(subtotals)

        json.dump({"total_objects:": total_all, "subtotal": subtotals}, f)
    # sys.exit(0)

    if split == "*":
        splits = ["val", "test", "train"]
    else:
        splits = [split]

    ############### GENERATE EMPTY root dataset and mini_val ########################
    for split in splits:
        dset = habitat.datasets.make_dataset("ObjectNav-v1")
        dset.category_to_task_category_id = category_to_task_category_id
        dset.category_to_mp3d_category_id = category_to_mp3d_category_id
        global_dset = f"{OUTPUT_JSON_FOLDER}/{split}/{split}.json{COMPRESSION}"
        if os.path.exists(global_dset):
            os.remove(global_dset)
        if not os.path.exists(os.path.dirname(global_dset)):
            os.mkdir(os.path.dirname(global_dset))
        # if split != "train":
        jsons_gz = glob.glob(
            f"{OUTPUT_JSON_FOLDER}/{split}/content/*.json{COMPRESSION}"
        )

        with mp_ctx.Pool(CPU_THREADS) as pool, tqdm.tqdm(
            total=len(jsons_gz), desc=f"Loading jsons for {split}"
        ) as pbar:
            for i, dset2 in enumerate(pool.map(read_dset, jsons_gz)):
                if i == 0 and split == "val":
                    MINI_SAMPLE = 10  # 30
                    print("Sampling %d episodes for minival" % MINI_SAMPLE)
                    min_dset_fname = f"{OUTPUT_JSON_FOLDER}/{split}_mini/{split}_mini.json{COMPRESSION}"
                    os.makedirs(os.path.dirname(min_dset_fname), exist_ok=True)

                    min_dset = habitat.datasets.make_dataset("ObjectNav-v1")
                    min_dset.category_to_task_category_id = (
                        category_to_task_category_id
                    )
                    min_dset.category_to_mp3d_category_id = (
                        category_to_mp3d_category_id
                    )

                    min_dset.episodes = list(
                        dset2.get_episode_iterator(
                            num_episode_sample=MINI_SAMPLE, cycle=False
                        )
                    )
                    save_dataset(min_dset, min_dset_fname)
                    # with gzip.open(min_dset_fname, "wt") as f:
                    #    f.write(min_dset.to_json())

                # dset.episodes.extend(dset2.episodes)
                pbar.update()
        save_dataset(dset, global_dset)
