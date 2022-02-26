import argparse
import habitat

from habitat import get_config as get_task_config
from habitat_baselines.il.disk_based.dataset.dataset import PickPlaceDataset


def generate_episode_dataset(config, mode):
    rearrangement_dataset = PickPlaceDataset(
        config,
        content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
        mode=mode,
        use_iw=config.IL.USE_IW,
        inflection_weight_coef=config.MODEL.inflection_weight_coef
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", type=str, default=""
    )
    parser.add_argument(
        "--episodes", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--mode", type=str, default="train"
    )
    args = parser.parse_args()

    config = habitat.get_config("habitat_baselines/config/object_rearrangement/il_pickplace_mp3d.yaml")

    cfg = config
    cfg.defrost()
    task_config = get_task_config(cfg.BASE_TASK_CONFIG_PATH)
    task_config.defrost()
    if args.task == "rearrangement":
        task_config.DATASET.TYPE = "PickPlaceDataset-v1"
        task_config.DATASET.DATA_PATH = args.episodes
        print("Episodes: {}".format(args.episodes))

    if len(args.scene) > 0:
        task_config.DATASET.CONTENT_SCENES = [args.scene]

    task_config.freeze()
    cfg.TASK_CONFIG = task_config
    cfg.freeze()

    generate_episode_dataset(cfg, args.mode, args.task, args.scene, args.use_semantic, args.no_vision)


if __name__ == "__main__":
    main()
