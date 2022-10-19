import glob
import os
import random
import argparse

from tqdm import tqdm
from scripts.utils.utils import write_json, write_gzip, load_dataset, write_gzip
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def get_habitat_sim_action_str(action):
    if action in ["TURN_RIGHT", "TURN_LEFT", "MOVE_FORWARD", "LOOK_UP", "LOOK_DOWN", "STOP"]:
        return action
    if action == HabitatSimActions.TURN_RIGHT:
        return "TURN_RIGHT"
    elif action == HabitatSimActions.TURN_LEFT:
        return "TURN_LEFT"
    elif action == HabitatSimActions.MOVE_FORWARD:
        return "MOVE_FORWARD"
    elif action == HabitatSimActions.LOOK_UP:
        return "LOOK_UP"
    elif action == HabitatSimActions.LOOK_DOWN:
        return "LOOK_DOWN"
    return "STOP"


def list_files(path):
    return glob.glob(path)


def merge_episodes(input_path_1, output_path):
    files = list_files(os.path.join(input_path_1, "*json.gz"))
    file_paths = [
        # "../objectnav_sem_exp/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fm_v3/train/content/",
        # "../objectnav_sem_exp/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fm_v2/train/content/",
        # "../objectnav_sem_exp/data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fm_v1/train/content/",
        "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fe_150k_balanced/train_50k/content/",
    ]

    count = 0
    for f in tqdm(files):
        scene_id = f.split("/")[-1]
        input_dataset_1 = load_dataset(f)

        for file_path in file_paths:
            input_dataset_2_path = os.path.join(file_path, scene_id)

            if os.path.exists(input_dataset_2_path):
                input_dataset_2 = load_dataset(input_dataset_2_path)

                input_dataset_1["episodes"].extend(input_dataset_2["episodes"])
            else:
                count += 1
        random.shuffle(input_dataset_1["episodes"])
        
        scene_output_path = os.path.join(output_path, scene_id.replace(".gz", ""))
        write_json(input_dataset_1, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)
    print("Missing {} scenes".format(count))


def convert_actions_to_commands(input_path, output_path):
    files = list_files(os.path.join(input_path, "*json.gz"))

    for f in tqdm(files):
        dataset = load_dataset(f)
        
        for episode in dataset["episodes"]:
            for step in episode["reference_replay"]:
                step["action"] = get_habitat_sim_action_str(step["action"])

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
        "--merge", action="store_true", dest="merge"
    )
    args = parser.parse_args()

    if args.merge:
        merge_episodes(args.input_path, args.output_path)
    else:
        convert_actions_to_commands(args.input_path, args.output_path)
