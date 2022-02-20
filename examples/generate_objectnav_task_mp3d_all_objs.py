#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import csv
import glob
import gzip
import itertools
import json
import lzma
import multiprocessing
import os
import os.path as osp
import pickle
import random
import sys
import time
import traceback
from collections import defaultdict

import GPUtil
import numpy as np
import pydash
import tqdm

import habitat
import habitat.datasets.eqa.mp3d_eqa_dataset as mp3d_dataset
import habitat_sim
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.object_nav import object_nav_dataset
from habitat.datasets.object_nav.create.object_nav_generator import (
    build_goal,
    generate_objectnav_episode,
)

# from habitat.datasets.object_nav.create.object_nav_generator_old import (
#     generate_new_obj_nav_episode,
# )
from habitat.datasets.pointnav.create.utils import get_mp3d_scenes
from habitat.datasets.utils import VocabDict, VocabFromText
from habitat.tasks.eqa.eqa import AnswerAction
from habitat.tasks.nav.nav import MoveForwardAction
from habitat.utils.test_utils import sample_non_stop_action

CFG_TEST = "configs/test/habitat_mp3d_eqa_test.yaml"
CLOSE_STEP_THRESHOLD = 0.028

COMPRESSION = ".gz"
OUTPUT_OBJ_FOLDER = (
    "data/datasets/objectnav_mp3d_v1/rest"
)
OUTPUT_JSON_FOLDER = "data/datasets/objectnav_mp3d_v1/rest/"
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
# Valid IDs
# print([mp3dcat40_mapinv[a] for a in mp3d_valid_ids])
# sys.exit(0)


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
        deviceId = 0 #i % NUM_GPUS
    else:
        deviceId = deviceIds[0]
    objnav_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
        deviceId
    )  # i % NUM_GPUS
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


def generate_scene(args):
    i, scene, split = args
    # TODO fix stools!
    # mp3d_valid_categories = np.array(list(set(name for name in mp3dcat_nametosynset.keys() if name in set(a.split('.')[0] for a in coco_map.values()))))
    objnav_config = get_objnav_config(i, scene)

    # if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
    #     objnav_config.DATASET
    # ):
    #     pytest.skip(
    #         "Please download Matterport3D EQA dataset to data folder."
    #     )

    # env = habitat.Env(config=objnav_config)
    # env.episodes[0].scene_id = new_test_scene
    # env.episodes[1].scene_id = new_test_scene
    # = env.sim

    # env.reset()
    sim = get_simulator(objnav_config)
    total_objects = len(sim.semantic_annotations().objects)
    # print(list(coco_map.values()))
    # Check there exists a navigable point
    test_point = sim.sample_navigable_point()
    if total_objects == 0 or not sim.is_navigable(np.array(test_point)):
        print("Scene is not navigable: %s" % scene)
        sim.close()
        return scene, total_objects, defaultdict(list), None

    objects = []

    # print(sim.semantic_annotations().objects[0].category.name(""))
    for source_id, source_obj in enumerate(
        tqdm.tqdm(
            sim.semantic_annotations().objects[1:],
            desc="Generating object data",
        )
    ):

        # print(dir(source_obj))
        # try:
        #    category_name = source_obj.category_name.name("")
        #    category_id
        # except AttributeError:
        category_name = copy.deepcopy(source_obj.category.name(""))
        category_id = source_obj.category.index("")

        if category_id not in mp3d_valid_ids:
            continue
        if category_name == None:
            print("ERROR NONE CATEGORY NAME: %s %d" % (scene, source_id))
            continue

            # if category_name not in mp3dcat_nametosynset:
            #    print('Skipping: %s' % category_name)
            #    continue
            # print(category_name)

            # print(mp3dcat_nametosynset[category_name])
            # print(mp3dcat_nametosynset[category_name])

        # if category_name not in mp3d_valid_categories:
        #    print('Skipping %s' % category_name)
        #    continue
        # print(str(source_obj.obb.center))
        # print(dir(source_obj.obb))
        # print((source_obj.obb.sizes))
        # getTransforms(source_obj.obb)
        # sys.exit(0)

        obj = {
            "center": source_obj.aabb.center,
            "id": int(source_obj.id.split("_")[-1]),
            "object_name": source_obj.id,
            "obb": source_obj.obb,
            "aabb": source_obj.aabb,
            "gravity_mobb": get_gravity_mobb(source_obj.obb),
            "category_id": category_id,
            "category_name": category_name,
        }
        objects.append(obj)

    print("Scene loaded.")
    scene_key = osp.basename((osp.dirname(scene)))
    fname_obj = f"{OUTPUT_OBJ_FOLDER}/{split}/content/{scene_key}_objs.pkl"
    fname = (
        f"{OUTPUT_JSON_FOLDER}/{split}/content/{scene_key}.json{COMPRESSION}"
    )

    if os.path.exists(fname_obj):
        with open(fname_obj, "rb") as f:
            goals_by_class = pickle.load(f)
        total_objects_by_cat = {k: len(v) for k, v in goals_by_class.items()}
    else:
        goals_by_class = defaultdict(list)
        cell_size = objnav_config.SIMULATOR.AGENT_0.RADIUS / 2.0
        for obj in tqdm.tqdm(objects, desc="Objects for %s:" % scene):
            print("Object id: %d" % obj["id"])
            print(obj["category_name"])

            goal = build_goal(
                sim,
                object_id=obj["id"],
                object_name_id=obj["object_name"],
                object_category_name=obj["category_name"],
                object_category_id=obj["category_id"],
                object_position=obj["center"],
                object_aabb=obj["aabb"],
                object_obb=obj["obb"],
                object_gmobb=obj["gravity_mobb"],
                cell_size=cell_size,
                grid_radius=3.0,
            )
            if goal == None:  # or len(goal.view_points) == 0:
                continue
            goals_by_class[obj["category_id"]].append(goal)
            # break
        os.makedirs(osp.dirname(fname_obj), exist_ok=True)
        total_objects_by_cat = {k: len(v) for k, v in goals_by_class.items()}
        with open(fname_obj, "wb") as f:
            pickle.dump(goals_by_class, f)

    if os.path.exists(fname):
        print("Scene already generated. Skipping")
        sim.close()
        return scene, total_objects, total_objects_by_cat, None

    # sim.close()
    # del sim
    # THE SIMULATOR MUST BE BLIND
    # objnav_config.defrost()
    # objnav_config.SIMULATOR.AGENT_0.SENSORS = []
    # objnav_config.freeze()
    # sim = get_simulator(objnav_config)

    if True:
        total_objs_json = 200 if split == "train" else 200
        # if split == 'val':
        #    total_obs_json = 100
        # if split == 'val_mini':
        #    total_objs_json = 30
        total_valid_objs = sum(total_objects_by_cat.values())
        dset = habitat.datasets.make_dataset("ObjectNav-v1")
        dset.category_to_task_category_id = category_to_task_category_id
        dset.category_to_mp3d_category_id = category_to_mp3d_category_id
        with tqdm.tqdm(total=total_objs_json, desc=scene) as pbar:

            for goals in goals_by_class.values():
                # TODO OPTIMIZE CREATE SIM WITHOUT SEMANTIC SENSOR SENSOR?
                # TODO OPTIMIZE IT TO SORT GOALS OF GEODESIC DISTANCE
                eps_generated = 0
                if True:
                    eps_per_obj = int(
                        len(goals) / total_valid_objs * total_objs_json + 0.5
                    )
                else:
                    eps_per_obj = 500
                try:
                    for ep in generate_objectnav_episode(
                        sim, goals, num_episodes=eps_per_obj
                    ):
                        dset.episodes.append(ep)
                        pbar.update()
                        eps_generated += 1
                except RuntimeError:
                    traceback.print_exc()
                    print("Skipping category")
                    pbar.update(eps_per_obj - eps_generated)
        print("Generation done")
        print("Start saving...")

        for ep in dset.episodes:
            # STRIP OUT PATH
            ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]
        
        print("Save at {}".format(fname))

        os.makedirs(osp.dirname(fname), exist_ok=True)
        save_dataset(dset, fname)
        # dset2 = habitat.datasets.make_dataset('ObjectNav-v1')
        # with gzip.open(fname, 'rt') as f:
        #     #dset2f.read()
        #     dset2.from_json(f.read())
        # assert dset2.category_vocab_dict_dense.word_list == dset.category_vocab_dict_dense.word_list
        # assert dset2.category_vocab_dict_sparse == dset.category_vocab_dict_sparse
    sim.close()
    return scene, total_objects, total_objects_by_cat, fname

    # for episode in generate_new_obj_nav_episode(env, [obj], num_episodes=1):
    #     print(episode)

    # list(map(lambda x:x.category.name(), sim.semantic_annotations().objects[1:]))
    # list(map(lambda x:(x.obb.sizes, x.obb.center), sim.semantic_annotations().objects[1:]))

    #
    # dataset = make_dataset(
    #     id_dataset=objnav_config.DATASET.TYPE, config=objnav_config.DATASET
    # )
    # # import random
    # # random.shuffle(dataset.episodes)
    # env = habitat.Env(config=objnav_config, dataset=dataset)
    # scenes = env._dataset.scene_ids
    # # env.episodes = [
    # #     episode
    # #     for episode in dataset.episodes
    # #     if int(episode.episode_id) in TEST_EPISODE_SET[:EPISODES_LIMIT]
    # # ]
    # print("Len episodes: ", len(env.episodes))
    # NUM_EPISODES = 40
    # from tqdm import tqdm
    # pbar = tqdm(total=len(env.episodes) * NUM_EPISODES)
    # env.episodes = [
    #     env.episodes[5]
    # ]
    # generator = eqa_generator.generate_new_start_eqa_episode(env)
    # for i in range(NUM_EPISODES):
    #     print(next(generator))


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
            dset = object_nav_dataset.ObjectNavDatasetV1.dedup_goals_dset(dset)
            f.write(dset.to_json())


def read_dset(json_fname):
    dset2 = habitat.datasets.make_dataset("ObjectNav-v1")
    file_opener = get_file_opener(json_fname)

    # compression = gzip if os.path.splitext(json_fname)[-1] == 'gz' else lzma
    with file_opener(json_fname, "rt") as f:
        # print(json_fname)
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
            scenes = get_mp3d_scenes(split)
        random.shuffle(scenes)
        scenes = [
            f"./data/scene_datasets/mp3d/{scene}/{scene}.glb"
            for scene in scenes
        ]
        blacklist = ["JmbYfDe2QKZ"]
        whitelist = ['SN83YJsR3w2'] #, '2n8kARJN3HM']
        #whitelist = ['SN83YJsR3w2'] #, 'SN83YJsR3w2', 'JmbYfDe2QKZ', 'VzqfbhrpDEA', '2n8kARJN3HM']
        # if split != 'val' assert
        scenes = [
            scene
            for scene in scenes
            if os.path.basename(os.path.dirname(scene)) in whitelist
        ]
        # Filtered scenes for train set
        # scenes = ['gTV8FGcVJC9', 'SN83YJsR3w2', 'JmbYfDe2QKZ', 'VzqfbhrpDEA', '2n8kARJN3HM']
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

    GPU_THREADS = NUM_GPUS * TASKS_PER_GPU
    print(GPU_THREADS)
    print("*" * 1000)
    CPU_THREADS = multiprocessing.cpu_count()
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
                    MINI_SAMPLE = 30
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
        # with get_file_opener(global_dset)(global_dset, 'w') as f:
        #    f.write('[]')
        # random.shuffle(dset.episodes) #This is really slow

        # save_dataset(dset, global_dset)
        # with gzip.open(global_dset, "wt") as f:
        #    f.write(dset.to_json())

        # print(subtotal)
        # print(subtotal_by_cat)

#     for scene in dataset.scene_ids:
#         env.sim.close()
#         del env._sim
#         del env
#         env = habitat.Env(config=objnav_config, dataset=dataset)
#         env.episodes = dataset.get_scene_episodes(scene)
#         for episode_num in range(len(env.episodes)):
#             # episode_count += 1
# #            if episode_count > 3584: #3584: #
#             env._current_episode_index = episode_num - 1
#             env.reset()
#             # print("env.current_episode: ", env.current_episode)
#             for episode in eqa_generator.generate_new_start_eqa_episode(env, NUM_EPISODES):
#                 if not episode:
#                     print("Episode id {} wasn't generated.".format(
#                         env.current_episode.episode_id))
#                     continue
#                 #episode.episode_id = episode_count
#                 episode_count += 1
#                 # print(episode)
#                 episodes.append(episode)
#                 pbar.update(1)
#         #         break
#         # break
#
#     dataset.episodes = episodes
#     dataset.to_json()
#
#     json_str = str(dataset.to_json())
#     scene_dataset_path = "data/datasets/eqa/mp3d/v2/{split}_extended/{" \
#                          "split}.json.gz".format(
#         split=objnav_config.DATASET.SPLIT)
#
#     import os
#     import gzip
#     if not os.path.exists(os.path.dirname(scene_dataset_path)):
#         os.makedirs(os.path.dirname(scene_dataset_path))
#
#     with gzip.GzipFile(scene_dataset_path, 'wb') as f:
#         f.write(json_str.encode("utf-8"))
#     print("Len of dataset episodes: ", len(dataset.episodes))
#     print("Dataset file: {}.".format(scene_dataset_path))
