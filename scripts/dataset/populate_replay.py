import argparse
import glob
import os

from scripts.utils.utils import load_dataset, load_json_dataset, write_json, write_gzip


def build_episode_replay_map(evaluation_meta):
    episode_replay_map = {}

    for episode in evaluation_meta:
        reference_replay = [{"action": action} for action in episode["actions"]]
        episode_replay_map[episode["episode_id"]] = {
            "success": episode["metrics"]["success"],
            "reference_replay": reference_replay
        }
    return episode_replay_map


def popualte_replay(input_path, dataset_path, output_path):
    evaluation_meta = load_json_dataset(input_path)
    episode_replay_map = build_episode_replay_map(evaluation_meta)

    fetch_episodes = ["4ok3usBNeis_25_[4.12308, -0.53553, 3.14021]_bed", "5cdEh9F2hJL_16_[5.32929, 0.0184, -4.36439]_sofa", "zt1RVoi7PcG_13_[5.34417, -2.88706, 9.01802]_bed", "svBbv1Pavdk_86_[4.25846, 0.07755, 3.8421]_sofa", "Nfvxx8J5NCo_66_[-5.42495, 0.18086, 1.02651]_tv_monitor", "cvZr5TUy5C5_70_[-7.86661, -2.92784, 5.69579]_bed"]

    files = glob.glob(os.path.join(dataset_path, "*json.gz"))
    for file in files:
        dataset = load_dataset(file)
        scene_id = file.split("/")[-1].replace(".gz", "")

        filtered_episodes = []
        for i, episode in enumerate(dataset["episodes"]):
            start_position = str(episode["start_position"])
            episode_id = "{}_{}_{}_{}".format(
                episode["scene_id"].split("/")[-1].split(".")[0], str(i), start_position, episode["object_category"]
            )
            if not episode_replay_map[episode_id]["success"] and episode_id in fetch_episodes:
                episode["reference_replay"] = episode_replay_map[episode_id]["reference_replay"]
                filtered_episodes.append(episode)
        
        dataset["episodes"] = filtered_episodes
        if len(filtered_episodes) == 0:
            continue
        scene_output_path = os.path.join(output_path, scene_id)
        print(scene_output_path, len(filtered_episodes))
        write_json(dataset, scene_output_path)
        write_gzip(scene_output_path, scene_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default=""
    )
    parser.add_argument(
        "--dataset-path", type=str, default=""
    )
    parser.add_argument(
        "--output-path", type=str, default=""
    )
    args = parser.parse_args()
    popualte_replay(args.input_path, args.dataset_path, args.output_path)

