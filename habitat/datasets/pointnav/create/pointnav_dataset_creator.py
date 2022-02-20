import os
from typing import Any, List, Optional

import numpy as np
from tqdm import tqdm

import habitat
from habitat.config.default import get_config
from habitat.core.dataset import Dataset
from habitat.datasets.pointnav.generator import generate_pointnav_episode
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal


class PointNavDatasetCreator(Dataset):
    r"""Point Navigation dataset.
    """
    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def _create_episode(
        episode_id,
        scene_id,
        start_position,
        start_rotation,
        target_position,
        shortest_path,
        info=None,
    ) -> Optional[NavigationEpisode]:
        episode_id = str(episode_id)
        goals = [NavigationGoal(position=target_position)]

        episode = NavigationEpisode(
            episode_id=episode_id,
            goals=goals,
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=start_rotation,
            shortest_paths=[shortest_path],
            info=info,
        )
        return episode

    @staticmethod
    def get_scenes(split, scene_path):
        raise NotImplementedError

    @staticmethod
    def check_config_paths_exist(config: Any) -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_path)

    def __init__(self, config: Any = None, scenes=None) -> None:
        self.episodes = []

        if config is None:
            return

        if not scenes:
            scenes = self.__class__.get_scenes(config.split, config.scene_path)
            # val_mini_scenes = self.__class__.get_scenes("val_mini",
            #                                            config.scene_path)
            # scenes = [scene for scene in scenes if scene not in val_mini_scenes]

        if config.num_houses > 0:
            scenes = scenes[: config.num_houses]
        progress_bar = tqdm(total=config.num_episodes)
        np.random.seed(config.seed)
        difficulty_counts = {}

        env_cfg = get_config()
        env_cfg.defrost()
        env_cfg.gpu_device_id = config.gpu_id
        env_cfg.task_name = "Nav-v0"
        env_cfg.freeze()

        for scene_path in scenes:
            env_cfg.defrost()
            env_cfg.SIMULATOR.SCENE = scene_path
            env_cfg.freeze()
            env = habitat.Env(config=env_cfg)
            env.seed(config.seed)

            num_episodes = round(config.num_episodes / len(scenes))
            # if
            # scene_path!=scenes[-1] else config.num_episodes - len(self.episodes)

            for episode in generate_pointnav_episode(env, num_episodes):
                episode.episode_id = len(self.episodes)
                self.episodes.append(episode)
                # dist_ratio = dataset.episodes[-1].info["distance_ratio"]
                # if dist_ratio < GEODESTIC_TO_EUCLID_RATIO:
                #     simple_episodes += 1

                progress_bar.update(1)
                # break
            env.close()
            del env

        print("scene_count: ", len(scenes))
        print("episodes_count: ", len(self.episodes))

    def to_json(self) -> str:
        result = habitat.core.utils.DatasetFloatJSONEncoder().encode(self)
        return result
