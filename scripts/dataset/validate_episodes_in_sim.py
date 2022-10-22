from collections import defaultdict
import math
import argparse
import habitat
import os
import glob
import numpy as np

from PIL import Image
from scripts.utils.utils import load_json_dataset, write_json
from tqdm import tqdm

config = habitat.get_config("configs/tasks/objectnav_mp3d_il.yaml")


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def validate_in_sim(
    cfg,
    output_path=None,
    scene_id=None,
):
    with habitat.Env(cfg) as env:
        num_episodes = len(env.episodes)
        invalid_episodes = 0
        
        for ep_id in tqdm(range(num_episodes)):
            env.reset()
            info = env.get_metrics()

            if info["distance_to_goal"] == np.inf or info["distance_to_goal"] == math.inf:
                invalid_episodes += 1
        print("Invalid episodes for scene: {} - {}".format(scene_id, invalid_episodes))
        return num_episodes, invalid_episodes


def iterate_scenes(cfg, input_path, output_path):
    files = glob.glob(os.path.join(input_path, "*json.gz"))
    invalid_count_map = defaultdict(int)
    total_episodes = defaultdict(int)

    failed_so_far = load_json_dataset(os.path.join(input_path, "done_so_far.json"))

    for path in files:
        if path in failed_so_far:
            continue
        cfg.defrost()
        cfg.DATASET.DATA_PATH = path
        cfg.freeze()
        print("Working on scene: {}".format(path))

        try:
            num_episodes, invalid_episodes = validate_in_sim(cfg, output_path, path.split("/")[-1])
            invalid_count_map[path] = invalid_episodes
            total_episodes[path] = num_episodes
        except Exception as e:
            print(path, e)
            failed_so_far.append(path)
            write_json(failed_so_far, os.path.join(input_path, "done_so_far.json"))
    
    print("invalid episodes: {}".format(invalid_count_map))
    print("all episodes: {}".format(total_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="demo"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL", "SUCCESS"] #, "TOP_DOWN_MAP"]
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS_WITHIN_1M = False
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD_V2 = False
    cfg.TASK.SIMPLE_REWARD.USE_DTG_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_ANGLE_SUCCESS_REWARD = True
    cfg.freeze()

    iterate_scenes(
        cfg,
        input_path=args.path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
