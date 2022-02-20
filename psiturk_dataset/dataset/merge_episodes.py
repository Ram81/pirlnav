import argparse
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gzip

from collections import defaultdict
from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset, load_vocab
from tqdm import tqdm


def merge_episodes(path, output_path):
    files = glob.glob(path + "*.json.gz")
    vocab = load_vocab()
    instructions = vocab["sentences"]

    dataset = {
        "episodes": [],
        "instruction_vocab": {
            "sentences": instructions
        }
    }

    for file_path in files:
        if "failed" in file_path:
            continue
        print("Loading episodes: {}".format(file_path))
        data = load_dataset(file_path)
        dataset["episodes"].extend(data["episodes"])

    print("Total episodes: {}".format(len(dataset["episodes"])))
    
    write_json(dataset, output_path)
    write_gzip(output_path, output_path)


def merge_objectnav_episodes(path, output_path):
    files = glob.glob(path + "*.json.gz")
    for file_path in files:
        print("Loading episodes: {}".format(file_path))
        scene_id = file_path.split("/")[-1]
        output_file_path = output_path + "{}".format(scene_id)
        print("output path: {}".format(output_file_path))
        source_data = load_dataset(file_path)
        dest_data = load_dataset(output_file_path)

        dest_data["episodes"].extend(source_data["episodes"])

        print("Total episodes: {}, Scene: {}".format(len(dest_data["episodes"]), scene_id))
        
        output_file_path = output_file_path.replace(".gz", "")
        write_json(dest_data, output_file_path)
        write_gzip(output_file_path, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_approvals/hits_max_length_1500.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sample_hits.json"
    )
    parser.add_argument(
        "--task", type=str, default="objectnav"
    )
    
    args = parser.parse_args()
    if args.task == "objectnav":
        merge_objectnav_episodes(args.input_path, args.output_path)
    else:
        merge_episodes(args.input_path, args.output_path)
    
