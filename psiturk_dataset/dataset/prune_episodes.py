import argparse
import gzip
import json
import random
import numpy as np

from tqdm import tqdm
from psiturk_dataset.utils.utils import load_dataset, load_json_dataset, write_gzip, write_json


def get_random_int(lb, ub):
    return random.randint(lb, ub)


def is_grab_release_action(step):
    if step.get("action") in ["GRAB_RELEASE", "grabReleaseObject"]:
        return True


def generate_closer_initialization(input_path, output_path, num_steps):
    data = load_dataset(input_path)

    avg_steps = 0
    avg_steps_before = 0
    for ep_id, episode in tqdm(enumerate(data["episodes"])):
        first_grab_action_index = 0
        for i, step in enumerate(episode["reference_replay"]):
            if is_grab_release_action(step) and step["is_grab_action"] and step["action_data"]["gripped_object_id"] != -1:
                first_grab_action_index = i - 5
                break
        
        # Get a random agent position closer to object
        episode_start_index = max(0, first_grab_action_index - num_steps)
        step_index = get_random_int(episode_start_index, first_grab_action_index)
        step = episode["reference_replay"][step_index]
        episode["start_position"] = step["agent_state"]["position"]
        episode["start_rotation"] = step["agent_state"]["rotation"]
        # episode["start_index"] = step_index

        # Modify replay buffer to start from intermediate step
        avg_steps_before += len(episode["reference_replay"])
        episode["reference_replay"][0]["agent_state"]["position"] = episode["start_position"]
        episode["reference_replay"][0]["agent_state"]["rotation"] = episode["start_rotation"]
        episode["reference_replay"] = [episode["reference_replay"][0]] + episode["reference_replay"][step_index:]

        avg_steps += len(episode["reference_replay"])

    print("Average number of steps before pruning: {}".format(avg_steps_before / len(data["episodes"])))
    print("Average number of steps after pruning: {}".format(avg_steps / len(data["episodes"])))

    write_json(data, output_path)
    write_gzip(output_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/datasets/object_rearrangement/v0/train/train_pruned.json"
    )
    parser.add_argument(
        "--num-steps", type=int, default=50
    )
    args = parser.parse_args()

    generate_closer_initialization(args.input_path, args.output_path, args.num_steps)


if __name__ == "__main__":
    main()
