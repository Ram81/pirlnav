import gzip
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode

import habitat
import habitat.datasets.pointnav.create.utils as utils
from habitat.config.default import get_config
from habitat.datasets.pointnav.create.pointnav_dataset_creator import (
    PointNavDatasetCreator,
)
from habitat.datasets.semantic_nav import generate_new_obj_nav_episode

# from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
# from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal

try:
    from _json import encode_basestring_ascii
except ImportError:
    encode_basestring_ascii = None
try:
    from _json import encode_basestring
except ImportError:
    encode_basestring = None


class DatasetFloatJSONEncoder(json.JSONEncoder):
    """
        JSON Encoder that set float precision for space saving purpose.
    """

    # Version of JSON library that encoder is compatible with.
    __version__ = "2.0.9"

    def default(self, object):
        return object.__dict__

    # Overriding method to inject own `_repr` function for floats with needed
    # precision.
    def iterencode(self, o, _one_shot=False):

        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(
            o,
            allow_nan=self.allow_nan,
            _repr=lambda x: format(x, ".5f"),
            _inf=float("inf"),
            _neginf=-float("inf"),
        ):
            if o != o:
                text = "NaN"
            elif o == _inf:
                text = "Infinity"
            elif o == _neginf:
                text = "-Infinity"
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: "
                    + repr(o)
                )

            return text

        _iterencode = json.encoder._make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)


class ObjectGoalNavDatasetCreatorV1(PointNavDatasetCreator):
    r"""Point Navigation dataset.
    """

    def generate_dataset(
        self,
        config: Any = None,
        scenes: List[str] = None,
        query_data: Dict = None,
    ) -> None:
        self.episodes = []

        if config is None:
            return

        dataset = EQDDatasetV1()

        if not scenes:
            scenes = self.__class__.get_scenes(config.split)
            # val_mini_scenes = self.__class__.get_scenes("val_mini",
            #                                            config.scene_path)
            # scenes = [scene for scene in scenes if scene not in val_mini_scenes]

        if config.num_houses > 0:
            scenes = scenes[: config.num_houses]
        progress_bar = tqdm(total=config.num_episodes)
        np.random.seed(config.seed)

        env_cfg = get_config("configs/tasks/eqd_create.yaml")
        env_cfg.defrost()
        env_cfg.gpu_device_id = config.gpu_id
        env_cfg.task_name = "Nav-v0"
        env_cfg.freeze()

        num_queries = sum([len(query_data[scene].items()) for scene in scenes])
        num_episodes_per_query = round(config.num_episodes / num_queries)

        for scene in scenes:
            if scene not in query_data:
                print(
                    "{} skipped as not presented in query data.".format(scene)
                )
                continue
            print(f"\nQueries from: {scene}")
            env_cfg.defrost()
            env_cfg.SIMULATOR.SCENE = config.scene_path.format(scene=scene)
            env_cfg.freeze()
            env = habitat.Env(config=env_cfg)
            env.seed(config.seed)
            env.episodes = [env.episodes[0]]
            env.episodes[0].scene_id = config.scene_path.format(scene=scene)
            env.reset()

            from time import time

            for query_id, query in enumerate(query_data[scene].items()):
                query[1]["type"] = "objects:room:level"
                print(f"Query id: {query_id}, {query[0]}, {query[1]}")
                t_gen_episode = time()
                for episode in generate_new_obj_nav_episode(
                    env, query, num_episodes_per_query
                ):
                    print(f"t_gen_episode: {time() - t_gen_episode }")
                    t_gen_episode = time()
                    episode.episode_id = len(dataset.episodes)
                    dataset.episodes.append(episode)

                    progress_bar.update(1)
                if len(dataset.episodes) > 1:
                    break
            env.close()
            del env

        print("scene_count: ", len(scenes))
        print("episodes_count: ", len(dataset.episodes))
        return dataset

    @staticmethod
    def get_scenes(split, scene_path="{scene}"):
        return utils.get_mp3d_scenes(split, scene_path)


def get_default_mp3d_v1_config(split: str = "train"):
    # "test_standard" test_challenge
    config = CfgNode()
    config.seed = 704184
    config.num_episodes = 30
    config.num_houses = 0
    config.split = split  # data/datasets/pointnav/gibson/v1
    config.output_path = "data/datasets/eqd/mp3d/v1/{}_30_eps/" "".format(
        config.split
    )

    config.IS_SINGLE_FILE = False
    config.OUTPUT_PATH_SINGLE_SCENE = "{}content/{{scene}}.json.gz".format(
        config.output_path
    )

    config.gpu_id = 0
    config.scene_path = "data/scene_datasets/mp3d/{scene}/{scene}.glb"
    return config


def main():
    config = get_default_mp3d_v1_config()
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if config.IS_SINGLE_FILE:
        dataset = ObjectGoalNavDatasetCreatorV1(config)
        json_str = str(dataset.to_json())

        main_dataset_file = "{}{}.json.gz".format(
            config.output_path, config.split
        )
        with gzip.GzipFile(main_dataset_file, "wb") as f:
            f.write(json_str.encode("utf-8"))

        print("Dataset file: {}".format(main_dataset_file))

    else:
        cur_config = config.clone()
        dataset = ObjectGoalNavDatasetCreatorV1(cur_config)
        json_str = str(dataset.to_json())

        with gzip.GzipFile(
            "{}{}.json.gz".format(config.output_path, config.split), "wb"
        ) as f:
            f.write(json_str.encode("utf-8"))

        print("Dataset file: {}".format(config.output_path))

        scenes = ObjectGoalNavDatasetCreatorV1.get_scenes(config.split)

        if config.num_houses > 0:
            scenes = scenes[: config.num_houses]
        progress_bar = tqdm(total=config.num_episodes)
        difficulty_counts = {}

        finished_episodes = 0
        for scene in scenes:
            scene_dataset_path = config.OUTPUT_PATH_SINGLE_SCENE.format(
                scene=scene
            )
            scene_path = config.scene_path.format(scene=scene)

            if os.path.exists(scene_dataset_path):
                print("{} exists.".format(scene_dataset_path))
                finished_episodes += round(config.num_episodes / len(scenes))
                continue

            cur_config.num_episodes = (
                round(config.num_episodes / len(scenes))
                if scene != scenes[-1]
                else config.num_episodes - finished_episodes
            )
            dataset = ObjectGoalNavDatasetCreatorV1(
                config=cur_config, scenes=[scene_path]
            )
            finished_episodes += len(dataset.episodes)
            json_str = str(dataset.to_json())

            if not os.path.exists(os.path.dirname(scene_dataset_path)):
                os.makedirs(os.path.dirname(scene_dataset_path))

            with gzip.GzipFile(scene_dataset_path, "wb") as f:
                f.write(json_str.encode("utf-8"))
            print("Dataset file: {}.".format(scene_dataset_path))
            progress_bar.update(len(dataset.episodes))


if __name__ == "__main__":
    main()
