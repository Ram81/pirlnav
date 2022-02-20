import argparse
import csv
import copy
import datetime
import glob
import gzip
import json
import re
import sys

from tqdm import tqdm
from habitat.datasets.utils import VocabFromText
from psiturk_dataset.dataset.check_train_val_leak import populate_points


max_instruction_len = 9
instruction_list = []
unique_action_combo_map = {}
max_num_actions = 0
num_actions_lte_tenk = 0
total_episodes = 0
excluded_ep = 0
task_episode_map = {}
VISITED_POINT_DICT = {}

def read_csv(path, delimiter=","):
    file = open(path, "r")
    reader = csv.reader(file, delimiter=delimiter)
    return reader


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def column_to_json(col):
    if col is None:
        return None
    return json.loads(col)


def get_csv_rows(csv_reader):
    rows = []
    for row in csv_reader:
        rows.append(row)
    return rows


def is_viewer_step(data):
    if "type" in data.keys():
        if data["type"] == "runStep" and data["step"] == "viewer":
            return True
    return False


def convert_to_episode(csv_reader, file_path):
    episode = {}
    viewer_step = False
    start_ts = 0
    end_ts = 0
    for row in csv_reader:
        uunique_id = row[0]
        step = row[1]
        timestamp = row[2]
        data = column_to_json(row[3])

        if start_ts == 0:
            start_ts = int(timestamp)

        if not viewer_step:
            viewer_step = is_viewer_step(data)

        if viewer_step:
            if data.get("event") == "setEpisode":
                scene_id = data["data"]["episode"]["sceneID"]
                episode_id = data["data"]["episode"]["episodeID"]

                unique_id = "{}:{}".format(scene_id, episode_id)

                if unique_id not in task_episode_map.keys():
                    task_episode_map[unique_id] = 0
                # else:
                #     print("Repeat episode: {}, File: {}, Unique Id: {}, Ep len: {}".format(unique_id, file_path, uunique_id, len(list(csv_reader))))
                task_episode_map[unique_id] += 1
                break


def replay_to_episode(replay_path, output_path, max_episodes=1):
    all_episodes = {
        "episodes": []
    }

    file_paths = glob.glob(replay_path + "/*.csv")
    for file_path in tqdm(file_paths):
        reader = read_csv(file_path)
        convert_to_episode(reader, file_path)

    if len(task_episode_map.keys()) > 0:
        print("Total episodes: {}".format(len(task_episode_map.keys())))
        count = 0
        print("Task episode map:\n")
        for key, v in task_episode_map.items():
            if v > 1:
                print(key)
                count += 1
        print("Duplicate episodes: {}".format(count))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-path", type=str, default="data/hit_data"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/data.json"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=1
    )
    args = parser.parse_args()
    replay_to_episode(args.replay_path, args.output_path, args.max_episodes)


if __name__ == '__main__':
    main()
