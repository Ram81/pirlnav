
import glob
import os
import random
import argparse

from collections import defaultdict
from tqdm import tqdm
from scripts.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset, write_gzip


def list_files(path):
    return glob.glob(path)


def sample_dataset(input_path, output_path):
    files = list_files(os.path.join(input_path, "*json.gz"))

    for f in tqdm(files):
        dataset = load_dataset(f)

        ep_per_cat = defaultdict(list)

        for episode in dataset["episodes"]:
            object_category = episode["object_category"]
            if len(ep_per_cat[object_category]) == 0:
                ep_per_cat[object_category].append(episode)
        
        episodes = []
        for key, value in ep_per_cat.items():
            episodes.extend(value)
        dataset["episodes"] = episodes
        print("Len episodes: {}".format(len(episodes)))

        scene_id = f.split("/")[-1]
        scene_output_path = os.path.join(output_path, scene_id.replace(".gz", ""))
        write_json(dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)


def copy_dataset(input_path, output_path):
    input_files = list_files(os.path.join(input_path, "*json.gz"))

    for f in tqdm(input_files):
        input_dataset = load_dataset(f)
        scene_output_path = os.path.join(output_path, f.split("/")[-1])
        output_dataset = load_dataset(scene_output_path)

        output_dataset["episodes"] = input_dataset["episodes"]
        print("Goals copy episodes: {}".format(len(output_dataset["episodes"])))

        scene_output_path = scene_output_path.replace(".gz", "")

        write_json(output_dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)


def copy_goals(input_path, output_path):
    input_files = list_files(os.path.join(input_path, "*json.gz"))

    for f in tqdm(input_files):
        input_dataset = load_dataset(f)
        scene_output_path = os.path.join(output_path, f.split("/")[-1])
        if not os.path.exists(scene_output_path):
            continue
        output_dataset = load_dataset(scene_output_path)

        output_dataset["goals_by_category"] = input_dataset["goals_by_category"]
        print("Goals copy episodes: {}".format(len(output_dataset["episodes"])))

        scene_output_path = scene_output_path.replace(".gz", "")

        write_json(output_dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)


def copy_category_mapping(input_path, output_path):
    input_dataset = load_dataset(input_path)
    files = list_files(os.path.join(output_path, "*json.gz"))

    for scene_output_path in tqdm(files):
        if not os.path.exists(scene_output_path):
            continue
        output_dataset = load_dataset(scene_output_path)

        output_dataset["category_to_task_category_id"] = input_dataset["category_to_task_category_id"]
        output_dataset["category_to_mp3d_category_id"] = input_dataset["category_to_mp3d_category_id"]
        output_dataset["gibson_to_mp3d_category_map"] = input_dataset["gibson_to_mp3d_category_map"]
        output_dataset["category_to_category_id"] = input_dataset["category_to_category_id"]
        print("Goals copy episodes: {}".format(len(output_dataset["episodes"])))

        scene_output_path = scene_output_path.replace(".gz", "")

        write_json(output_dataset, scene_output_path)
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
        "--copy", dest="copy", action="store_true"
    )
    parser.add_argument(
        "--copy-goals", dest="copy_goals", action="store_true"
    )
    parser.add_argument(
        "--copy-cat-map", dest="copy_cat_map", action="store_true"
    )
    args = parser.parse_args()

    if args.copy:
        copy_dataset(args.input_path, args.output_path)
    elif args.copy_goals:
        copy_goals(args.input_path, args.output_path)
    elif args.copy_cat_map:
        copy_category_mapping(args.input_path, args.output_path)
    else:
        sample_dataset(args.input_path, args.output_path)
