import argparse
import cv2
import habitat
import json
import sys
import time
import os

from habitat import Config, logger
from habitat import get_config as get_task_config
from habitat_baselines.rearrangement.dataset.episode_dataset import RearrangementEpisodeDataset
from habitat_baselines.objectnav.dataset.episode_dataset import (
    ObjectNavEpisodeDataset,
    ObjectNavEpisodeDatasetV2,
    ObjectNavEpisodeDatasetV3
)

from time import sleep

from PIL import Image

objectnav_scene_splits = {
    "split_1": ['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '5LpN3gDmAk7', '5q7pvUzZiYa', '759xd9YjKW5', '7y3sRwLe3Va', '82sE5b5pLXE', '8WUmhLawc2A', 'B6ByNegPMKs', 'D7G3Y4RVNrH', 'D7N2EKCX4Sj', 'E9uDoFAP3SH'],
    "split_2": ['EDJbREhghzL', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'JF19kD82Mey', 'JeFG25nYj2p', 'PX4nDJXEHrG', 'Pm6F8kyY3z2', 'PuKPg4mmafe', 'S9hNv5qa7GM', 'ULsKaCPVFJR', 'Uxmj2M2itWa', 'V2XKFyX4ASd', 'VFuaQ6m2Qom', 'VLzqgDo317F'],
    "split_3": ['VVfe2KiqLaN', 'Vvot9Ly1tCj', 'XcA2TqTSSAj', 'YmJkqBEsHnH', 'ZMojNkEp431', 'aayBHfsNo7d', 'ac26ZMwG7aT', 'b8cTxDM8gDG', 'cV4RVeZvu5T', 'dhjEzFoUFzH', 'e9zR4mvMWw7', 'gZ6f7yhEvPG', 'i5noydFURQK', 'jh4fc5c5qoQ'],
    "split_4": ['kEZ7cmS4wCh', 'mJXqzFtmKg4', 'p5wJjkQkbXX', 'pRbA3pwrgk9', 'qoiz87JEwZ2', 'r1Q1Z4BcV1o', 'r47D5H71a5s', 'rPc6DW4iMge', 's8pcmisQ38h', 'sKLMLpTHeUy', 'sT4fr6TAbpF', 'uNb9QFRL6hY', 'ur6pFq6Qu1A', 'vyrNrziPKCB'],
    "split_5": ['17DRP5sb8fy', 'mJXqzFtmKg4']
}

def generate_episode_dataset(config, mode, task, split_name="split_1", use_semantic=False, no_vision=False):
    if task == "rearrangement":
        rearrangement_dataset = RearrangementEpisodeDataset(
            config,
            content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
            mode=mode,
            use_iw=config.IL.USE_IW,
            inflection_weight_coef=config.MODEL.inflection_weight_coef
        )
    else:
        if use_semantic:
            logger.info("Using RBGD + semantic dataset")
            dataset = ObjectNavEpisodeDatasetV2(
                config,
                content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
                mode=mode,
                use_iw=config.IL.USE_IW,
                split_name=split_name,
                inflection_weight_coef=config.MODEL.inflection_weight_coef
            )
        elif no_vision:
            logger.info("Using no vision dataset")
            dataset = ObjectNavEpisodeDatasetV3(
                config,
                content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
                mode=mode,
                use_iw=config.IL.USE_IW,
                split_name=split_name,
                inflection_weight_coef=config.MODEL.inflection_weight_coef
            )
        else:
            logger.info("Using RGBD dataset")
            rearrangement_dataset = ObjectNavEpisodeDataset(
                config,
                content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
                mode=mode,
                use_iw=config.IL.USE_IW,
                split_name=split_name,
                inflection_weight_coef=config.MODEL.inflection_weight_coef
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", type=str, default="empty_house"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--mode", type=str, default="train"
    )
    parser.add_argument(
        "--task", type=str, default="rearrangement"
    )
    parser.add_argument(
        "--use-semantic", dest='use_semantic', action='store_true'
    )
    parser.add_argument(
        "--no-vision", dest='no_vision', action='store_true'
    )
    args = parser.parse_args()

    config = habitat.get_config("habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml")
    if args.task == "objectnav":
        config = habitat.get_config("habitat_baselines/config/objectnav/il_objectnav.yaml")

    cfg = config
    cfg.defrost()
    task_config = get_task_config(cfg.BASE_TASK_CONFIG_PATH)
    task_config.defrost()
    if args.task == "rearrangement":
        task_config.DATASET.TYPE = "RearrangementDataset-v1"
        task_config.DATASET.DATA_PATH = args.episodes
        print("Episodes: {}".format(args.episodes))

    task_config.DATASET.CONTENT_SCENES = [args.scene]
    if args.task == "objectnav":
        task_config.ENVIRONMENT.MAX_EPISODE_STEPS = 1800
        task_config.DATASET.TYPE = "ObjectNav-v2"
        # task_config.DATASET.DATA_PATH = args.episodes
        task_config.DATASET.CONTENT_SCENES = objectnav_scene_splits[args.scene]
    task_config.freeze()
    cfg.TASK_CONFIG = task_config
    cfg.freeze()

    observations = generate_episode_dataset(cfg, args.mode, args.task, args.scene, args.use_semantic, args.no_vision)

if __name__ == "__main__":
    main()
