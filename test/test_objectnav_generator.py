#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import habitat
from habitat import make_dataset
from habitat.config.default import get_config
from habitat.core.dataset import Dataset, Episode, ObjectInScene, SceneState
from habitat.core.simulator import AgentState
from habitat.datasets.object_nav.create.create_objectnav_dataset_with_added_objects import (
    category_to_mp3d_category_id,
    category_to_task_category_id,
)
from habitat.datasets.pointnav.create.utils import (
    get_gibson_scenes,
    get_habitat_gibson_scenes,
    get_habitat_mp3d_scenes,
    get_mp3d_scenes,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

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


def test_objnav_val_split_sample():
    CFG_TEST = "configs/test/habitat_mp3d_object_nav_test.yaml"
    objnav_config = get_config(CFG_TEST)
    objnav_config.defrost()
    objnav_config.DATASET.SPLIT = "val"
    objnav_config.DATASET.DATA_PATH = "/private/home/agokaslan/ai_habitat/object_nav_dataset/data/datasets/objectnav/mp3d/v1_floatencoded_cats_90/{split}/{split}.json.gz"
    objnav_config.freeze()
    dataset = habitat.make_dataset(
        id_dataset=objnav_config.DATASET.TYPE, config=objnav_config.DATASET
    )
    import copy

    ds500 = copy.copy(dataset)
    ep_iter = dataset.get_episode_iterator(num_episode_sample=500, cycle=False)
    ds500.epsiodes = list(ep_iter)

    ds30 = copy.copy(dataset)
    ds30.epsiodes = ds30.epsiodes[:10]

    # with gzip.GzipFile("data/datasets/objectnav/mp3d/v3/val/val.json.gz", 'wb') as f:
    #     f.write(ds500.to_json().encode("utf-8"))

    # with gzip.GzipFile("data/datasets/objectnav/mp3d/v3/mini_val/mini_val.json.gz", 'wb') as f:
    #     f.write(ds30.to_json().encode("utf-8"))


def test_objnav_episode_generator():
    CFG__OBJECT_NAV = "configs/test/habitat_all_sensors_test.yaml"
    objnav_config = get_config(CFG__OBJECT_NAV)
    objnav_config.defrost()
    objnav_config.DATASET.SPLIT = "train"
    # objnav_config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    # objnav_config.TASK.SENSORS.append("HEADING_SENSOR")
    new_test_scene = "/private/home/maksymets/data/gibson/gibson_tiny_manual_verified/Allensville.glb"
    objnav_config.SIMULATOR.SCENE = new_test_scene
    objnav_config.freeze()

    # if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
    #     objnav_config.DATASET
    # ):
    #     pytest.skip(
    #         "Please download Matterport3D EQA dataset to data folder."
    #     )

    env = habitat.Env(config=objnav_config)
    env.episodes[0].scene_id = new_test_scene
    env.episodes[1].scene_id = new_test_scene
    sim = env.sim
    env.reset()
    # sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.SIMULATOR)
    objects = []
    for source_id, source_obj in enumerate(
        sim.semantic_annotations().objects[1:]
    ):
        obj = {
            "center": source_obj.obb.center,
            "id": source_id + 1,
            "category_name": source_obj.category.name(""),
        }
        objects.append(obj)
    print("Scene loaded.")
    from habitat.datasets.object_nav.create.object_nav_generator import (
        build_goal,
    )

    gen_goals = 0
    for obj in objects:
        print(f"Object id: {obj['id']}.")
        res = build_goal(
            sim,
            object_id=obj["id"],
            object_name_id=obj["category_name"],
            object_position=obj["center"],
        )
        if res:
            gen_goals += 1
        # for episode in generate_new_obj_nav_episode(env, [obj], num_episodes=1):
        #     print(episode)

    print(f"Total goals generated: {gen_goals}")
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

    episodes = []
    episode_count = 0


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


def _gen_objnav_episode(config):
    episode = ObjectGoalNavEpisode(
        episode_id="0",
        scene_id=config.SIMULATOR.SCENE,
        scene_state=SceneState(
            objects=[
                ObjectInScene(
                    object_id="0",
                    object_template="data/assets/objects/ycb_google_16k/configs_gltf/004_sugar_box",
                    position=[
                        2.8853018283843994,
                        0.26378023624420166,
                        0.09014610946178436,
                    ],
                    rotation=[
                        0.0,
                        0.3386494815349579,
                        0.0,
                        0.9409126043319702,
                    ],
                )
            ]
        ),
        start_position=[-3.0133917, 0.04623024, 7.3064547],
        start_rotation=[0, 0.163276, 0, 0.98658],
        object_category="bed",
        goals=[
            ObjectGoal(
                object_id="0",
                object_category="food",
                position=[
                    2.8853018283843994,
                    0.26378023624420166,
                    0.09014610946178436,
                ],
                view_points=[
                    ObjectViewLocation(
                        agent_state=AgentState(
                            position=[
                                3.4509520530700684,
                                0.26378023624420166,
                                -0.31346720457077026,
                            ]
                        )
                    )
                ],
            )
        ],
        info={"geodesic_distance": 0.001},
    )
    return episode


def test_objnav_episode_generator_with_added_objects():
    CFG__OBJECT_NAV = "configs/tasks/obj_nav_mp3d.yaml"
    objnav_config = get_config(CFG__OBJECT_NAV)
    objnav_config.defrost()
    objnav_config.DATASET.SPLIT = "val"
    # objnav_config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    # objnav_config.TASK.SENSORS.append("HEADING_SENSOR")
    new_test_scene = "/private/home/maksymets/data/gibson/gibson_tiny_manual_verified/Allensville.glb"
    objnav_config.SIMULATOR.SCENE = new_test_scene
    objnav_config.freeze()

    # if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
    #     objnav_config.DATASET
    # ):
    #     pytest.skip(
    #         "Please download Matterport3D EQA dataset to data folder."
    #     )

    env = habitat.Env(config=objnav_config)
    env.episodes[0].scene_id = new_test_scene
    env.episodes[1].scene_id = new_test_scene
    sim = env.sim
    env.reset()
    objects = []


def test_clean_objnav_dataset_with_added_objects():
    CFG__OBJECT_NAV = "configs/tasks/objectnav_mp3d_256.yaml"
    objnav_config = get_config(CFG__OBJECT_NAV)
    objnav_config.defrost()
    objnav_config.DATASET.SPLIT = "train"
    objnav_config.DATASET.CONTENT_SCENES = ["*"]
    objnav_config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
    objnav_config.TASK.DISTANCE_TO_GOAL.GEO_DIST_RADIUS = 0.8
    objnav_config.TASK.SUCCESS_DISTANCE = 0.8
    objnav_config.TASK.SUCCESS.SUCCESS_DISTANCE = 0.8
    objnav_config.TASK.MEASUREMENTS = [
        "DISTANCE_TO_GOAL",
        "SUCCESS",
        "SPL",
        "SOFT_SPL",
        "COLLISIONS",
    ]
    objnav_config.TASK.OBJECTSEMANTIC_SENSOR.WIDTH = 256
    objnav_config.TASK.OBJECTSEMANTIC_SENSOR.HEIGHT = 256

    objnav_config.DATASET.DATA_PATH = "data/datasets/objectnav/mp3d/objnav_ycb_objs_200k_single_clean/{split}/{split}.json.gz"
    objnav_config.freeze()

    env = habitat.Env(config=objnav_config)
    valid_episodes = []
    import tqdm

    for _ in tqdm.tqdm(range(len(env.episodes))):
        env.reset()
        obs = env.step(env.task.action_space.sample())
        metrics = env.get_metrics()
        # print(metrics["distance_to_goal"])
        # print(env.current_episode.info["geodesic_distance"])
        if metrics["distance_to_goal"] < 990:
            valid_episodes.append(env.current_episode)
            # print(f"Episode ok")
        # else:
        #     print(f"Episode invalid")
        # if len(valid_episodes) > 5:
        #     break
    env._dataset.episodes = valid_episodes
    import gzip

    fname = "data/datasets/objectnav/mp3d/objnav_ycb_objs_200k_single_clean_2_stage/{split}/{split}.json.gz".format(
        split=objnav_config.DATASET.SPLIT
    )
    with gzip.open(fname, "wt") as f:
        f.write(env._dataset.to_json())

    env.close()


def _construct_dataset(episodes_per_scene, scenes):
    episodes = []
    for scene_idx in range(len(scenes)):
        for i in range(episodes_per_scene):
            episode = Episode(
                episode_id=str(len(episodes)),
                scene_id=scenes[scene_idx],
                start_position=[0, 0, 0],
                start_rotation=[0, 0, 0, 1],
            )
            episodes.append(episode)
    dataset = Dataset()
    dataset.episodes = episodes
    return dataset


def test_create_episode_per_scene_objnav_dataset_with_added_objects():
    CFG__OBJECT_NAV = "configs/tasks/objectnav_mp3d_256.yaml"
    objnav_config = get_config(CFG__OBJECT_NAV)
    objnav_config.defrost()
    objnav_config.TASK.OBJECTSEMANTIC_SENSOR.WIDTH = 256
    objnav_config.TASK.OBJECTSEMANTIC_SENSOR.HEIGHT = 256
    objnav_config.TASK.TYPE = "ObjectNavAddedObj-v1"
    objnav_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    objnav_config.TASK.OBJ_GEN_FAR_DIST = 35.0
    objnav_config.freeze()
    split = "train"
    mp3d_scenes = get_mp3d_scenes(split)
    disabled_scenes = {
        "VLzqgDo317F",
        "5q7pvUzZiYa",
        "EDJbREhghzL",
        "VLzqgDo317F",
        "YmJkqBEsHnH",
        "dhjEzFoUFzH",
        "B6ByNegPMKs",
        "E9uDoFAP3SH",
        # "SN83YJsR3w2",
        "VzqfbhrpDEA",
        "YmJkqBEsHnH",
        "dhjEzFoUFzH",
        "gTV8FGcVJC9",
        "gZ6f7yhEvPG",
        "r1Q1Z4BcV1o",
        "ur6pFq6Qu1A",
        "vyrNrziPKCB",
    }
    print(list(filter(lambda x: not x in disabled_scenes, mp3d_scenes)))
    mp3d_scenes = [
        f"./data/scene_datasets/mp3d/{scene}/{scene}.glb"
        for scene in mp3d_scenes
    ]
    habitat_mp3d_scenes = [
        f"./data/scene_datasets/habitat_matterport/{scene}"
        for scene in get_habitat_mp3d_scenes("*")
    ]

    episodes_per_scene = 2
    dataset = _construct_dataset(
        episodes_per_scene=episodes_per_scene, scenes=habitat_mp3d_scenes
    )
    dataset.category_to_task_category_id = category_to_task_category_id
    dataset.category_to_mp3d_category_id = category_to_mp3d_category_id

    env = habitat.Env(config=objnav_config, dataset=dataset)

    import tqdm

    for _ in tqdm.tqdm(range(len(env.episodes))):
        try:
            env.reset()
            obs = env.step(env.task.action_space.sample())

            print(
                f" env.current_episode.scene_id = {env.current_episode.scene_id}"
            )
        except BaseException:
            print(
                f"failed env.current_episode.scene_id = {env.current_episode.scene_id}"
            )
    env.close()


def test_create_episode_per_scene_objnav_dataset():
    CFG__OBJECT_NAV = "configs/tasks/objectnav_mp3d_256.yaml"
    objnav_config = get_config(CFG__OBJECT_NAV)
    objnav_config.defrost()
    objnav_config.TASK.OBJECTSEMANTIC_SENSOR.WIDTH = 256
    objnav_config.TASK.OBJECTSEMANTIC_SENSOR.HEIGHT = 256
    objnav_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    split = "train"
    mp3d_scenes = get_mp3d_scenes(split)
    enabled_scenes = {
        # "VLzqgDo317F",
        "5q7pvUzZiYa",
        "EDJbREhghzL",
        "VLzqgDo317F",
        # "YmJkqBEsHnH",
        # "dhjEzFoUFzH",
        # "B6ByNegPMKs",
        # "E9uDoFAP3SH",
        # "SN83YJsR3w2",
        # "VzqfbhrpDEA",
        # "YmJkqBEsHnH",
        # "dhjEzFoUFzH",
        # "gTV8FGcVJC9",
        # "gZ6f7yhEvPG",
        # "r1Q1Z4BcV1o",
        # "ur6pFq6Qu1A",
        # "vyrNrziPKCB",
    }
    objnav_config.DATASET.CONTENT_SCENES = list(enabled_scenes)
    objnav_config.freeze()

    dataset = make_dataset(
        id_dataset=objnav_config.DATASET.TYPE, config=objnav_config.DATASET
    )
    # get one episode per scene
    dataset.episodes = [
        np.random.choice(dataset.get_scene_episodes(scene), size=1)[0]
        for scene in dataset.scene_ids
    ]

    mp3d_scenes = list(filter(lambda x: x in enabled_scenes, mp3d_scenes))
    print(mp3d_scenes)
    mp3d_scenes = [
        f"./data/scene_datasets/mp3d/{scene}/{scene}.glb"
        for scene in mp3d_scenes
    ]
    # mp3d_scenes = mp3d_scenes[32:]
    gibson_scenes = get_gibson_scenes(split)
    gibson_scenes = [
        f"./data/scene_datasets/gibson/{scene}.glb" for scene in gibson_scenes
    ]

    habitat_mp3d_scenes = [
        f"./data/scene_datasets/habitat_matterport/{scene}"
        for scene in get_habitat_mp3d_scenes("*")
    ]

    # episodes_per_scene = 2
    # dataset = _construct_dataset(
    #     episodes_per_scene=episodes_per_scene, scenes=mp3d_scenes
    # )
    # dataset.category_to_task_category_id = category_to_task_category_id
    # dataset.category_to_mp3d_category_id = category_to_mp3d_category_id

    env = habitat.Env(config=objnav_config, dataset=dataset)

    import tqdm

    for _ in tqdm.tqdm(range(len(env.episodes))):
        try:
            env.reset()
            obs = env.step(env.task.action_space.sample())

            print(
                f" env.current_episode.scene_id = {env.current_episode.scene_id}"
            )
        except BaseException:
            print(
                f"failed env.current_episode.scene_id = {env.current_episode.scene_id}"
            )
    env.close()


def test_mp3d_objnav_dataset():
    CFG__OBJECT_NAV = "configs/tasks/objectnav_mp3d_256.yaml"
    objnav_config = get_config(CFG__OBJECT_NAV)
    objnav_config.defrost()
    enabled_scenes = {
        "VLzqgDo317F",
        "5q7pvUzZiYa",
        "EDJbREhghzL",
        "VLzqgDo317F",
        "YmJkqBEsHnH",
        "dhjEzFoUFzH",
        "B6ByNegPMKs",
        "E9uDoFAP3SH",
        # "SN83YJsR3w2",
        # "VzqfbhrpDEA",
        "YmJkqBEsHnH",
        "dhjEzFoUFzH",
        # "gTV8FGcVJC9",
        "gZ6f7yhEvPG",
        "r1Q1Z4BcV1o",
        "ur6pFq6Qu1A",
        "vyrNrziPKCB",
    }
    objnav_config.DATASET.CONTENT_SCENES = list(enabled_scenes)
    objnav_config.freeze()

    dataset = make_dataset(
        id_dataset=objnav_config.DATASET.TYPE, config=objnav_config.DATASET
    )
    # get one episode per scene
    dataset.episodes = [
        np.random.choice(dataset.get_scene_episodes(scene), size=5)[0]
        for scene in dataset.scene_ids
    ]

    env = habitat.Env(config=objnav_config, dataset=dataset)

    import tqdm

    for _ in tqdm.tqdm(range(len(env.episodes))):
        env.reset()
        obs = env.step(env.task.action_space.sample())
    env.close()


if __name__ == "__main__":
    test_mp3d_objnav_dataset()