import argparse
import copy

from tqdm import tqdm
from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset


def append_start_step(path, output_path, task):
    data = load_dataset(path)
    prev_trajectory_length = 0
    new_trajectory_length = 0

    for episode in tqdm(data["episodes"]):
        prev_trajectory_length += len(episode["reference_replay"])
        if episode["reference_replay"][0]["action"] == "STOP":
            continue
        start_step = copy.deepcopy(episode["reference_replay"][0])
        start_step["action"] = "STOP"

        sensor_data = start_step["agent_state"]["sensor_data"]
        for key, value in sensor_data.items():
            sensor_data[key] = {
                "rotation": [0, 0, 0, 1],
                "translation": [0, 1.5, 0]
            }

        start_step["agent_state"] = {
            "position": episode["start_position"],
            "rotation": episode["start_rotation"],
            "sensor_data": sensor_data
        }
        reference_replay = [start_step] + episode["reference_replay"]
        episode["reference_replay"] = reference_replay
        new_trajectory_length += len(episode["reference_replay"])
    

    print("No start step trajectory length: {}".format(prev_trajectory_length))
    print("With start step trajectory length: {}".format(new_trajectory_length))
    
    write_json(data, output_path)
    write_gzip(output_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default=""
    )
    parser.add_argument(
        "--output-path", type=str, default=""
    )
    parser.add_argument(
        "--task", type=str, default="objectnav"
    )
    args = parser.parse_args()

    append_start_step(args.input_path, args.output_path, args.task)

