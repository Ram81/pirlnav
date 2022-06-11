import argparse
import glob
import os

from scripts.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset, write_gzip

def filter_dataset(src_dataset_path, sample_dataset_path, output_path):
    files = glob.glob(os.path.join(src_dataset_path, "*json.gz"))
    for file in files:
        src_dataset = load_dataset(file)
        scene_id = file.split("/")[-1].replace(".gz", "")
        sample_episodes_path = os.path.join(sample_dataset_path, "{}.gz".format(scene_id))

        if not os.path.exists(sample_episodes_path):
            print("Missing episodes: {}".format(sample_episodes_path))
            continue
        sample_dataset = load_dataset(sample_episodes_path)

        total_episodes = len(src_dataset["episodes"])

        start_position_dict = {}
        for episode in sample_dataset["episodes"]:
            start_position = str(episode["start_position"])
            start_position_dict[start_position] = episode["object_category"]
        
        filtered_episodes = []
        for episode in src_dataset["episodes"]:
            start_position = str(episode["start_position"])
            if start_position_dict.get(start_position) == episode["object_category"]:
                continue
            filtered_episodes.append(episode)

        src_dataset["episodes"] = filtered_episodes
        path = os.path.join(output_path, "{}".format(scene_id))

        filtered_episodes_len = len(filtered_episodes)
        print("Writing at {} - {}".format(path, total_episodes - filtered_episodes_len))
        write_json(src_dataset, path)
        write_gzip(path, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dataset-path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--sample-dataset-path", type=str, default="demo"
    )
    parser.add_argument(
        "--output-path", type=str, default="demo"
    )
    args = parser.parse_args()

    filter_dataset(args.src_dataset_path, args.sample_dataset_path, args.output_path)
        