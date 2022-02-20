import argparse
import cv2
import habitat
import json
import sys
import time
import os

from habitat import Config
from habitat import get_config as get_task_config
from habitat_baselines.rearrangement.dataset.goal_dataset import RearrangementGoalDatasetV2

from time import sleep

from PIL import Image

config = habitat.get_config("habitat_baselines/config/object_rearrangement/ppo_agile_object_rearrangement.yaml")


def generate_episode_dataset(cfg, mode):
    rearrangement_dataset = RearrangementGoalDatasetV2(
        cfg,
        content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
        mode=mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", type=str, default="empty_house"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/datasets/object_rearrangement/v3/train/train.json.gz"
    )
    parser.add_argument(
        "--mode", type=str, default="train_goals"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    task_config = get_task_config("configs/tasks/rearrangement_video.yaml")
    task_config.defrost()
    task_config.DATASET.DATA_PATH = args.episodes
    task_config.DATASET.CONTENT_SCENES = [args.scene]
    task_config.freeze()
    cfg.TASK_CONFIG = task_config
    cfg.freeze()

    observations = generate_episode_dataset(cfg, args.mode)

if __name__ == "__main__":
    main()
