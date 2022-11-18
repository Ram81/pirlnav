import argparse

from scripts.utils.utils import load_json_dataset
from tqdm import tqdm


def compute_behavior_metrics(path, only_on_success=False, only_on_failure=False):
    dataset = load_json_dataset(path)

    behavior_metrics = {
        "peeks": 0,
        "panoramic_turns": 0,
        "beelines": 0,
        "exhaustive_search": 0,
        "sight_coverage": 0,
        "occupancy_coverage": 0,
    }

    num_episodes = len(dataset)
    for episode_metrics in tqdm(dataset):
        if only_on_success and not episode_metrics["metrics"]["success"]:
            continue
        if only_on_failure and episode_metrics["metrics"]["success"]:
            continue
        behavior_metrics["peeks"] += int(len(episode_metrics["behavior_metrics"]["room_revisitation_map_strict"].keys()) > 0)
        behavior_metrics["panoramic_turns"] += int(episode_metrics["behavior_metrics"]["panoramic_turns_strict"] > 0)
        if episode_metrics["metrics"]["success"]:
            behavior_metrics["beelines"] += int(episode_metrics["behavior_metrics"]["beeline"])
        behavior_metrics["exhaustive_search"] += int(episode_metrics["behavior_metrics"]["sight_coverage"] > 0.75)
        behavior_metrics["sight_coverage"] += episode_metrics["metrics"]["exploration_metrics.sight_coverage"]
        behavior_metrics["occupancy_coverage"] += episode_metrics["metrics"]["exploration_metrics.coverage"]

    for key, value in behavior_metrics.items():
        behavior_metrics[key] = behavior_metrics[key] / num_episodes
    
    print(behavior_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--success-only", dest="only_on_success", action="store_true"
    )
    parser.add_argument(
        "--failure-only", dest="only_on_failure", action="store_true"
    )
    args = parser.parse_args()

    compute_behavior_metrics(args.path, args.only_on_success, args.only_on_failure)
