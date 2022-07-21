import argparse
import os
import glob

from scripts.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset, write_gzip
from tqdm import tqdm


def count_episodes(path):
    files = glob.glob(os.path.join(path, "*json.gz"))
    episodes = 0

    for f in tqdm(files):
        dataset = load_dataset(f)
        episodes += len(dataset["episodes"])

    print("Total episodes: {}".format(episodes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    args = parser.parse_args()

    count_episodes(args.path)
