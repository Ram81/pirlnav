import glob
import os
import random
import argparse

from collections import defaultdict
from tqdm import tqdm
from scripts.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset, write_gzip


def list_files(path):
    return glob.glob(path)


def sample_dataset(input_path, output_path, episodes_per_scene=100, clear_replay=False):
    files = list_files(os.path.join(input_path, "*json.gz"))

    for f in tqdm(files):
        dataset = load_dataset(f)
        sampled_episodes = random.sample(dataset["episodes"], min(len(dataset["episodes"]), episodes_per_scene))

        # Clear replay buffer
        if clear_replay:
            for episode in sampled_episodes:
                if episode.get("reference_replay"):
                    episode["reference_replay"] = []
                    del episode["reference_replay"]
                    del episode["attempts"]
                    del episode["scene_dataset"]
                    del episode["scene_state"]
                    del episode["is_thda"]

        dataset["episodes"] = sampled_episodes

        scene_id = f.split("/")[-1]
        scene_output_path = os.path.join(output_path, scene_id.replace(".gz", ""))
        write_json(dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)


def sample_filtered_dataset(input_path, output_path, episodes_per_scene=100, clear_replay=False, sample_dataset_path=None):
    #sample_dataset_path = ["data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_77k/train/content/", "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_s_path/train/content/"]
    sample_dataset_path = ["data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fm/train/content/", "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/train_sample_4k_random/content/", "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/train_sample_4k_unseen/content/"]
    total_existing_episodes = 0
    VISITED_POINT_DICT = {}
    for sample_path in sample_dataset_path:
        files = list_files(os.path.join(sample_path, "*json.gz"))
        print("Sampling from: {}, {}".format(sample_path, len(files)))

        for f in files:
            dataset = load_dataset(f)
            scene_id = f.split("/")[-1].split(".")[0]
            if scene_id not in VISITED_POINT_DICT.keys():
                VISITED_POINT_DICT[scene_id] = {}

            for episode in dataset["episodes"]:
                start_position = str(episode["start_position"])
                if start_position not in VISITED_POINT_DICT[scene_id].keys():
                    VISITED_POINT_DICT[scene_id][start_position] = 1
                total_existing_episodes += 1

    print("Total existing episodes: {}".format(total_existing_episodes))

    files = list_files(os.path.join(input_path, "*json.gz"))

    for f in files:
        dataset = load_dataset(f)
        scene_id = f.split("/")[-1].split(".")[0]
        filtered_episodes = []

        for episode in dataset["episodes"]:
            start_position = str(episode["start_position"])
            if VISITED_POINT_DICT[scene_id].get(start_position) != 1:
                filtered_episodes.append(episode)

        print("Num filtered episodes: {}".format(len(dataset["episodes"]) - len(filtered_episodes)))

        sampled_episodes = random.sample(filtered_episodes, episodes_per_scene)

        # Clear replay buffer
        if clear_replay:
            for episode in sampled_episodes:
                if episode.get("reference_replay"):
                    episode["reference_replay"] = []
                    del episode["reference_replay"]
                    del episode["attempts"]
                    del episode["scene_dataset"]
                    del episode["scene_state"]
                    del episode["is_thda"]

        dataset["episodes"] = sampled_episodes

        scene_id = f.split("/")[-1]
        scene_output_path = os.path.join(output_path, scene_id.replace(".gz", ""))
        write_json(dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/datasets/objectnav/objectnav_hm3d/"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/datasets/objectnav/objectnav_hm3d/"
    )
    parser.add_argument(
        "--sample-dataset-path", type=str, default="data/datasets/objectnav/objectnav_hm3d/"
    )
    parser.add_argument(
        "--episodes-per-scene", type=int, default=100
    )
    parser.add_argument(
        "--clear-replay", dest="clear_replay", action="store_true"
    )
    parser.add_argument(
        "--filter", dest="filter", action="store_true"
    )
    args = parser.parse_args()

    if args.filter:
        sample_filtered_dataset(args.input_path, args.output_path, args.episodes_per_scene, args.clear_replay)
    else:
        sample_dataset(args.input_path, args.output_path, args.episodes_per_scene, args.clear_replay)
