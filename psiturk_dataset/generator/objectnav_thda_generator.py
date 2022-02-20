import argparse
import attr
import habitat
import os
import numpy as np
import sys
import tqdm

from collections import defaultdict
from habitat import get_config
from habitat.core.dataset import Dataset, Episode
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.datasets.pointnav.create.utils import get_mp3d_scenes, get_habitat_mp3d_scenes
from habitat.datasets.object_nav.create.create_objectnav_dataset_with_added_objects import (
    category_to_mp3d_category_id,
    category_to_task_category_id,
)
from psiturk_dataset.utils.utils import load_dataset, write_json, write_gzip

def get_scenes(split):
    mp3d_scenes = get_mp3d_scenes(split)
    disabled_scenes = {
        "17DRP5sb8fy",
    }
    filtered_scenes = list(filter(lambda x: not x in disabled_scenes, mp3d_scenes))
    return filtered_scenes

def read_config(path):
    filtered_scenes = get_scenes("train")
    objnav_config = get_config(path)
    objnav_config.defrost()
    objnav_config.TASK.TYPE = "ObjectNavAddedObj-v1"
    objnav_config.TASK.SENSORS = ['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']
    objnav_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
    objnav_config.DATASET.TYPE = "ObjectNav-v1"
    objnav_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    objnav_config.TASK.OBJ_GEN_NEAR_DIST = 15.0
    objnav_config.TASK.OBJ_GEN_FAR_DIST = 35.0
    objnav_config.TASK.NUM_UNIQ_SELECTED_OBJECTS = 5
    # objnav_config.DATASET.CONTENT_SCENES = filtered_scenes
    objnav_config.freeze()
    return objnav_config


def _construct_dataset(episodes_per_scene, scenes):
    episodes = []
    for scene_idx in range(len(scenes)):
        for i in range(episodes_per_scene):
            episode = ObjectGoalNavEpisode(
                episode_id=str(len(episodes)),
                scene_id=scenes[scene_idx],
                start_position=[0, 0, 0],
                start_rotation=[0, 0, 0, 1],
                goals=[],
            )
            episodes.append(episode)
    dataset = Dataset()
    dataset.episodes = episodes
    return dataset


def get_episode_json(episode):
    episode._shortest_path_cache = None
    for goal in episode.goals:
        for view_point in goal.view_points:
            if isinstance(view_point.agent_state.position, np.ndarray):
                view_point.agent_state.position = view_point.agent_state.position.tolist()
    return attr.asdict(episode)


def generate_objectnav_dataset_with_inserted_objects(
    objnav_config, split, output_path, num_episodes
):
    mp3d_scenes = get_mp3d_scenes(split)
    disabled_scenes = {
        '2n8kARJN3HM', 'JmbYfDe2QKZ', 'SN83YJsR3w2', 'gTV8FGcVJC9', 'VzqfbhrpDEA', '17DRP5sb8fy', 'GdvgFV5R1Z5',
        'HxpKQynjfin', 'gZ6f7yhEvPG', 'ur6pFq6Qu1A', 'Pm6F8kyY3z2', '17DRP5sb8fy', 'D7G3Y4RVNrH', 'GdvgFV5R1Z5', 'B6ByNegPMKs', 'dhjEzFoUFzH', 'V2XKFyX4ASd'
    }

    filtered_scenes = list(filter(lambda x: not x in disabled_scenes, mp3d_scenes))
    print(filtered_scenes)
    print("Scenes after filtering: {}".format(len(filtered_scenes)))
    print("All scenes: {}".format(mp3d_scenes))
    mp3d_scenes = [
        f"./data/scene_datasets/mp3d/{scene}/{scene}.glb"
        for scene in filtered_scenes
    ]

    episodes_per_scene = int(num_episodes / len(filtered_scenes))

    dataset = _construct_dataset(
        episodes_per_scene=episodes_per_scene, scenes=mp3d_scenes
    )
    dataset.category_to_task_category_id = category_to_task_category_id
    dataset.category_to_mp3d_category_id = category_to_mp3d_category_id

    print("Total episodes: {}".format(len(dataset.episodes)))

    scene_episode_map = defaultdict(list)

    env = habitat.Env(config=objnav_config, dataset=dataset)

    for _ in tqdm.tqdm(range(len(env.episodes))):
        env.reset()
        obs = env.step(env.task.action_space.sample())

        print(
            f" env.current_episode.scene_id = {env.current_episode.scene_id}"
        )
    env.close()

    dataset_json = {
        "episodes": [],
        "category_to_task_category_id": dataset.category_to_task_category_id,
        "category_to_mp3d_category_id": dataset.category_to_mp3d_category_id,
    }

    for episode in dataset.episodes:
        scene_id = episode.scene_id.split("/")[-1]
        episode_json = get_episode_json(episode)
        episode_json["scene_id"] = "mp3d/{}/{}".format(scene_id.split(".")[0], scene_id)
        episode_json["is_thda"] = True

        if hasattr(episode, "goals") and len(episode.goals) > 0:
            scene_episode_map[scene_id].append(episode_json)

    output_train_gz_path = "/".join(output_path.split("/")[:-1]) + "/train.json"
    write_json(dataset_json, output_train_gz_path)
    write_gzip(output_train_gz_path, output_train_gz_path)

    objectnav_dataset_path = "data/datasets/objectnav_mp3d_v1/train/content/{}.json.gz"
    for scene, episodes in scene_episode_map.items():
        scene = scene.split("/")[-1].split(".")[0]
        episode_data = load_dataset(objectnav_dataset_path.format(scene))
        episode_data["category_to_task_category_id"] = dataset.category_to_task_category_id
        episode_data["category_to_mp3d_category_id"] = dataset.category_to_mp3d_category_id

        print("Loaded {} episodes for scene {}".format(len(episode_data["episodes"]), scene))
        if len(episodes) == 0:
            continue
        episode_data["episodes"] = episodes

        path = output_path + "/{}.json".format(scene)
        write_json(episode_data, path)
        write_gzip(path, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-episodes", type=str, default="data/datasets/objectnav_mp3d_v1/train/content/"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/datasets/objectnav_mp3d_thda/train/content"
    )
    parser.add_argument(
        "--split", type=str, default="train"
    )
    parser.add_argument(
        "--config", type=str, default="configs/tasks/objectnav_mp3d_il.yaml"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1000
    )
    args = parser.parse_args()
    cfg = read_config(args.config)

    generate_objectnav_dataset_with_inserted_objects(cfg, args.split, args.output_path, args.num_episodes)


