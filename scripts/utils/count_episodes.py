import argparse
from collections import defaultdict
import os
import glob

from scripts.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset, write_gzip
from tqdm import tqdm


def count_episodes(path):
    files = glob.glob(os.path.join(path, "*json.gz"))
    episodes = 0
    object_categories = defaultdict(int)

    for f in tqdm(files):
        dataset = load_dataset(f)
        episodes += len(dataset["episodes"])

        print(dataset["category_to_task_category_id"])

        for episode in dataset["episodes"]:
            object_categories[episode["object_category"]] += 1

        print(f.split("/")[-1], len(dataset["episodes"]))

    print("Total episodes: {}".format(episodes))
    print(object_categories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    args = parser.parse_args()

    count_episodes(args.path)
