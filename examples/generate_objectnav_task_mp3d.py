#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pydash
import tqdm

import time

import habitat
import habitat.datasets.eqa.mp3d_eqa_dataset as mp3d_dataset
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.semantic_nav.obj_nav import object_nav_dataset
from habitat.datasets.semantic_nav.obj_nav.create.obj_nav_generator import (
    generate_new_obj_nav_episode,
)
from habitat.datasets.semantic_nav.obj_nav.create.obj_nav_generator_replica import (
    build_goal,
    generate_objectnav_episode,
)
from habitat.tasks.eqa.eqa import AnswerAction
from habitat.tasks.nav.nav import MoveForwardAction
from habitat.utils.test_utils import sample_non_stop_action
from collections import defaultdict

CFG_TEST = "configs/test/habitat_mp3d_eqa_test.yaml"
CLOSE_STEP_THRESHOLD = 0.028

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
mp3dcat_nametosynset = dict()
with open('examples/mpcat40.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for line in reader:
        id_ = int(line[0].strip())
        for synset in line[3].strip().split(','):
            mp3dcat40_map[synset] = id_
        mp3dcat_nametosynset[line[1].strip()] = synset

#print(mp3dcat_nametosynset)

with open("examples/coco_to_synset_edited.json", "r") as f:
    coco_map = json.load(f)

mp3d_valid_ids = np.array(list(set(mp3dcat40_map[synset] for synset in coco_map.values())))

mapping_dict = dict()
mapping_dict[mp3dcat40_map['stool.n.01']] = mp3dcat40_map['chair.n.01']


def generate_scene(args):
    i, scene = args
    #TODO fix stools!
    mp3d_valid_categories = np.array(list(set(name for name in mp3dcat_nametosynset.keys() if name in set(a.split('.')[0] for a in coco_map.values()))))

    CFG_EQA = "configs/test/habitat_all_sensors_test.yaml"
    objnav_config = get_config(CFG_EQA).clone()
    objnav_config.defrost()
    #objnav_config.SIMULATOR_GPU_ID = i % 2
    objnav_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = i % 2
    objnav_config.DATASET.DATA_PATH = './data/datasets/pointnav/mp3d/v1/val/val.json.gz'
    objnav_config.DATASET.SCENES_DIR = './data/scene_datasets/'
    objnav_config.DATASET.SPLIT = "train"
    # objnav_config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    # objnav_config.TASK.SENSORS.append("HEADING_SENSOR")
    #new_test_scene = "/private/home/maksymets/data/gibson/gibson_tiny_manual_verified/" + tiny_gibson_scenes[0] + '.glb'
    new_test_scene = scene#'data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb'
    objnav_config.SIMULATOR.SCENE = new_test_scene
    objnav_config.freeze()

    # if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
    #     objnav_config.DATASET
    # ):
    #     pytest.skip(
    #         "Please download Matterport3D EQA dataset to data folder."
    #     )

    #env = habitat.Env(config=objnav_config)
    #env.episodes[0].scene_id = new_test_scene
    #env.episodes[1].scene_id = new_test_scene
    # = env.sim

 
    #env.reset()
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.SIMULATOR)
    objects = []
    total_objects = len(sim.semantic_annotations().objects)
    #print(list(coco_map.values()))
    #Check there exists a navigable point
    test_point = sim.sample_navigable_point()
    if not sim.is_navigable(np.array(test_point)):
        print('Scene is not navigable: %s' % scene)
        return scene, total_objects, defaultdict(list) 

    #print(sim.semantic_annotations().objects[0].category.name(""))
    for source_id, source_obj in enumerate(
        sim.semantic_annotations().objects
        ):
        #print(dir(source_obj))
        #try:
        #    category_name = source_obj.category_name.name("")
        #    category_id
        #except AttributeError:
        category_name = source_obj.category.name('')
        category_id = source_obj.category.index('')

        if category_id in mapping_dict:
            category_id = mapping_dict[category_id]

        if category_id not in mp3d_valid_ids:
            print('Skipping: %s' % category_name)
            continue

            #if category_name not in mp3dcat_nametosynset:
            #    print('Skipping: %s' % category_name)
            #    continue
            #print(category_name)
            
            #print(mp3dcat_nametosynset[category_name])
            #print(mp3dcat_nametosynset[category_name])

        #if category_name not in mp3d_valid_categories:
        #    print('Skipping %s' % category_name)
        #    continue
        #print(str(source_obj.obb.center))
        obj = {
            "center": source_obj.obb.center,
            "id": source_id + 1,
            "category_id" : category_id,
            "category_name": category_name,
        }
        objects.append(obj)
    
    print("Scene loaded.")
    import pickle
    scene_key = osp.basename((osp.dirname(scene)))
    fname_obj = f"./data/datasets/objectnav/mp3d/v1/train/content/{scene_key}_objs.pkl"
    fname = f"./data/datasets/objectnav/mp3d/v1/train/content/{scene_key}.json.gz"



    if os.path.exists(fname_obj):
        with open(fname_obj, 'rb') as f:
            goals_by_class = pickle.load(f)
        total_objects_by_cat = {k:len(v) for k, v in goals_by_class.items()}
        

    else:
        goals_by_class = defaultdict(list)

        for obj in tqdm.tqdm(objects, desc='Objects for %s:' % scene):
            print("Object id: %d" % obj["id"])
            print(obj["category_name"])
            goal = build_goal(
                sim,
                object_id=obj["id"],
                object_name_id=obj["category_name"],
                object_position=obj["center"],
                cell_size=0.5,
                grid_radius= 3.0,
            )
            if goal == None:# or len(goal.view_points) == 0:
                continue
            goals_by_class[obj["category_id"]].append(goal)
            #break

        total_objects_by_cat = {k:len(v) for k, v in goals_by_class.items()}

    os.makedirs(osp.dirname(fname_obj), exist_ok=True)
    if os.path.exists(fname):
        print('Scene already generated. Skipping')
        return scene, total_objects, total_objects_by_cat
    with open(fname_obj, 'wb') as f:
        pickle.dump(goals_by_class, f)
    if True:
        eps_per_obj = 500
        dset = habitat.datasets.make_dataset("ObjectNav-v1")
        
        with tqdm.tqdm(total=len(goals_by_class) * eps_per_obj) as pbar:
            for goals in goals_by_class.values():
                #TODO OPTIMIZE CREATE SIM WITHOUT SEMANTIC SENSOR SENSOR?
                try:
                    for ep in generate_objectnav_episode(
                        sim, goals, num_episodes=eps_per_obj
                    ):
                        dset.episodes.append(ep)
                        pbar.update()
                except RuntimeError:
                    pbar.update()

        for ep in dset.episodes:
            #STRIP OUT PATH
            ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]


        os.makedirs(osp.dirname(fname), exist_ok=True)
        with gzip.open(fname, "wt") as f:
            f.write(dset.to_json())

    sim.close()
    return scene, total_objects, total_objects_by_cat

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

   

if __name__ == '__main__':
    mp_ctx = multiprocessing.get_context("fork")

    new_test_scene = "/private/home/maksymets/data/gibson/gibson_tiny_manual_verified/" + tiny_gibson_scenes[0] + '.glb'
    scenes = glob.glob('./data/scene_datasets/mp3d/*/*.glb')
    a = dict()
    np.random.seed(1234)
    with mp_ctx.Pool(16) as pool, tqdm.tqdm(total=len(scenes)) as pbar, open('train_subtotals.json', 'w') as f:
        for scene, subtotal, subtotal_by_cat in pool.imap_unordered(generate_scene, enumerate(scenes)):
            a[scene]= (subtotal, subtotal_by_cat)
            print('*' * 10)
            print(a)
        json.dump(a, f)
            #print(subtotal)
            #print(subtotal_by_cat)

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
