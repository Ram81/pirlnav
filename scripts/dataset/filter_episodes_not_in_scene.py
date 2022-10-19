import glob
import os
import json
import argparse

from tqdm import tqdm
from scripts.utils.utils import write_json, write_gzip, load_dataset, write_gzip


def list_files(path):
    return glob.glob(path)


def filter_episodes_not_in_scene(input_path, output_path):
    files = list_files(os.path.join(input_path, "*json.gz"))

    filtered_eps = 0
    for f in tqdm(files):
        dataset = load_dataset(f)
        goals = [ g.split("_")[1] for g in list(dataset["goals_by_category"].keys())]
        for i, g in enumerate(goals):
            if goals[i] == "tv":
                goals[i] = "tv_monitor"
        
        episodes = []
        set_objects = []
        for episode in dataset["episodes"]:
            set_objects.append(episode["object_category"])
            if episode["object_category"] not in goals:
                filtered_eps += 1
                continue
            episodes.append(episode)
        print("Filtered episodes: {} - {} - {}".format(filtered_eps, goals, set(set_objects)))

        dataset["episodes"] = episodes
        scene_id = f.split("/")[-1]
        scene_output_path = os.path.join(output_path, scene_id.replace(".gz", ""))
        write_json(dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)
    print("Filtered episodes: {}".format(filtered_eps))


def goal_matching(input_path):
    files = list_files(os.path.join(input_path, "*json.gz"))

    for f in tqdm(files):
        scene_id = f.split("/")[-1]
        dataset = load_dataset(f)
        goals = [g.split("_")[1] for g in list(dataset["goals_by_category"].keys())]
        src_dataset = os.path.join("data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fe_100k_balanced/train/content/", scene_id)
        src_dataset = load_dataset(src_dataset)
        print(scene_id, dataset.keys())
        goals = []
        src_goals = []
        for goal in src_dataset["goals_by_category"].keys():
            goals.append(json.dumps(src_dataset["goals_by_category"][goal]))
            src_goals.append(json.dumps(src_dataset["goals_by_category"][goal]))
        diff = set(src_goals).difference(set(goals))
        print(len(diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/datasets/objectnav/objectnav_hm3d/"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/datasets/objectnav/objectnav_hm3d/"
    )
    parser.add_argument(
        "--match", action="store_true", dest="match"
    )
    args = parser.parse_args()

    if args.match:
        goal_matching(args.input_path)
    else:
        filter_episodes_not_in_scene(args.input_path, args.output_path)
