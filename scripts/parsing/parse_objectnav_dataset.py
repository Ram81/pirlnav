import argparse
import csv
import copy
import datetime
import glob
import gzip
import json
import os
import zipfile

from collections import defaultdict
from tqdm import tqdm
from scripts.utils.utils import load_dataset


max_instruction_len = 9
instruction_list = []
unique_action_combo_map = {}
max_num_actions = 0
num_actions_lte_tenk = 0
total_episodes = 0
excluded_ep = 0
task_episode_map = defaultdict(list)

def read_csv(path, delimiter=","):
    file = open(path, "r")
    reader = csv.reader(file, delimiter=delimiter)
    return reader


def read_csv_from_zip(archive, path, delimiter=","):
    file = archive.open(path)
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


def preprocess(instruction):
    tokens = instruction.split()
    if len(tokens) < max_instruction_len:
        tokens = tokens + ["<pad>"] * (max_instruction_len - len(tokens))
    return " ".join(tokens)


def append_instruction(instruction):
    instruction_list.append(instruction)


def get_object_states(data):
    object_states = []
    for object_state in data["objectStates"]:
        object_states.append({
            "object_id": object_state["objectId"],
            "translation": object_state["translation"],
            "rotation": object_state["rotation"],
            "motion_type": object_state["motionType"],
        })
    return object_states


def get_action(data):
    if data is None:
        return None
    return data.get("action")


def is_physics_step(action):
    return (action == "stepPhysics")


def remap_action(action):
    if action == "turnRight":
        return "TURN_RIGHT"
    elif action == "turnLeft":
        return "TURN_LEFT"
    elif action == "moveForward":
        return "MOVE_FORWARD"
    elif action == "moveBackward":
        return "MOVE_BACKWARD"
    elif action == "lookUp":
        return "LOOK_UP"
    elif action == "lookDown":
        return "LOOK_DOWN"
    elif action == "grabReleaseObject":
        return "GRAB_RELEASE"
    elif action == "stepPhysics":
        return "NO_OP"
    return "STOP"


def handle_step(step, episode, unique_id, timestamp):
    if step.get("event"):
        if step["event"] == "setEpisode":
            data = copy.deepcopy(step["data"]["episode"])
            task_episode_map[data["scene_id"]].append(int(data["episode_id"]))

            episode["episode_id"] = unique_id # data["episode_id"]
            episode["scene_id"] = data["scene_id"]
            if len(episode["scene_id"].split("/")) == 1:
                episode["scene_id"] = "mp3d/{}/{}".format(episode["scene_id"].split(".")[0], episode["scene_id"])
            if "gibson" in episode["scene_id"]:
                episode["scene_id"] = "gibson_semantic/{}/{}".format(episode["scene_id"].split("/")[-1].split(".")[0], episode["scene_id"].split("/")[-1])
            episode["start_position"] = data["startState"]["position"]
            episode["start_rotation"] = data["startState"]["rotation"]
            episode["object_category"] = data["object_category"]
            episode["start_room"] = data.get("start_room")
            episode["shortest_paths"] = data.get("shortest_paths")
            episode["is_thda"] = data.get("is_thda")
            episode["info"] = data.get("info")
            episode["scene_dataset"] = data.get("scene_dataset")
            episode["scene_state"] = data.get("scene_state")
            episode["goals"] = []
            if episode["is_thda"]:
                episode["goals"] = data["goals"]

            episode["reference_replay"] = []

        elif step["event"] == "handleAction":
            action = remap_action(step["data"]["action"])
            data = step["data"]
            replay_data = {
                "action": action
            }
            replay_data["agent_state"] = {
                "position": data["agentState"]["position"],
                "rotation": data["agentState"]["rotation"],
                "sensor_data": data["agentState"]["sensorData"]
            }
            episode["reference_replay"].append(replay_data)

    elif step.get("type"):
        if step["type"] == "finishStep":
            return True
    return False


def convert_to_episode(csv_reader):
    episode = {}
    viewer_step = False
    start_ts = 0
    end_ts = 0
    for row in csv_reader:
        unique_id = row[0]
        step = row[1]
        timestamp = row[2]
        data = column_to_json(row[3])

        if start_ts == 0:
            start_ts = int(timestamp)

        if not viewer_step:
            viewer_step = is_viewer_step(data)

        if viewer_step:
            is_viewer_step_finished = handle_step(data, episode, unique_id, timestamp)
        end_ts = int(timestamp)

    # Append start and stop action
    start_action = copy.deepcopy(episode["reference_replay"][0])
    start_action["action"] = "STOP"
    stop_action = copy.deepcopy(episode["reference_replay"][-1])
    stop_action["action"] = "STOP"
    episode["reference_replay"] = [start_action] + episode["reference_replay"] + [stop_action]
    actual_episode_length = len(episode["reference_replay"])

    start_dt = datetime.datetime.fromtimestamp(start_ts / 1000)
    end_dt = datetime.datetime.fromtimestamp(end_ts / 1000)
    hit_duration = (end_dt - start_dt).total_seconds()

    episode_length = {
        "actual_episode_length": actual_episode_length,
        "hit_duration": hit_duration
    }
    return episode, episode_length


def replay_to_episode(
    replay_path, output_path, max_episodes=16,  max_episode_length=1000, sample=False, append_dataset=False, is_thda=False,
    is_gibson=False
):
    all_episodes = {
        "episodes": []
    }

    episode_lengths = []
    start_pos_map = {}
    episodes = []
    file_paths = glob.glob(replay_path + "/*.csv")
    scene_episode_map = defaultdict(list)
    duplicates = 0
    for file_path in tqdm(file_paths):
        reader = read_csv(file_path)

        episode, counts = convert_to_episode(reader)

        episode_key = str(episode["start_position"]) + "_{}".format(episode["scene_id"])

        if episode_key in start_pos_map.keys():
            duplicates += 1
            continue
        start_pos_map[episode_key] = 1

        if not is_gibson and "gibson" in episode["scene_id"]:
            continue
        if is_thda and not episode.get("is_thda"):
            continue
        if not is_thda and episode.get("is_thda"):
            continue

        if len(episode["reference_replay"]) <= max_episode_length:
            scene_episode_map[episode["scene_id"]].append(episode)
            all_episodes["episodes"].append(episode)
            episode_lengths.append(counts)
        if sample:
            if len(episode_lengths) >= max_episodes:
                break
    print("Total duplicate episodes: {}".format(duplicates))

    objectnav_dataset_path = "data/datasets/objectnav_mp3d_v1/train/content/{}.json.gz"
    if "val" in output_path:
        objectnav_dataset_path = objectnav_dataset_path.replace("train", "val")
        print("Using val path")
    if is_thda:
        objectnav_dataset_path = "data/datasets/objectnav_mp3d_thda/train/content/{}.json.gz"
    if is_gibson:
        objectnav_dataset_path = "../Object-Goal-Navigation/data/datasets/objectnav/gibson/v1.1/train_generated/content/{}.json.gz"
    for scene, episodes in scene_episode_map.items():
        scene = scene.split("/")[-1].split(".")[0]
        # print(objectnav_dataset_path)
        if not os.path.isfile(objectnav_dataset_path.format(scene)):
            print("Source dataset missing: {}".format(scene))
            continue
        episode_data = load_dataset(objectnav_dataset_path.format(scene))
        episode_data["episodes"] = episodes

        path = output_path + "/{}.json".format(scene)

        if append_dataset:
            existing_episodes = load_dataset(path + ".gz")
            len_before = len(episode_data["episodes"])
            episode_data["episodes"].extend(existing_episodes["episodes"])
            print("Appending new episodes to existing scene: {} -- {} -- {}".format(scene, len_before, len(episode_data["episodes"])))
        print(path)
        write_json(episode_data, path)
        write_gzip(path, path)


def show_average(all_episodes, episode_lengths):
    print("Total episodes: {}".format(len(all_episodes["episodes"])))

    total_episodes = len(all_episodes["episodes"])
    total_hit_duration = 0

    total_actions = 0
    num_eps_gt_than_2k = 0
    for episode_length  in episode_lengths:
        total_hit_duration += episode_length["hit_duration"]
        total_actions += episode_length["actual_episode_length"]
        num_eps_gt_than_2k += 1 if episode_length["actual_episode_length"] > 1900 else 0

    print("\n\n")
    print("Average hit duration")
    print("All hits: {}, Total duration: {}, Num episodes: {}".format(round(total_hit_duration / total_episodes, 2), total_hit_duration, total_episodes))
    
    print("\n\n")
    print("Average episode length:")
    print("All Hits: {}, Num actions: {}, Num episodes: {}".format(round(total_actions / total_episodes, 2), total_actions, total_episodes))

    print("\n\n")
    print("Episodes greater than 1.9k actions: {}".format(num_eps_gt_than_2k))


def list_missing_episodes():
    missing_episode_data = {}
    for key, categories in task_episode_map.items():
        object_category_map = defaultdict(int)
        for object_category in categories:
            object_category_map[object_category] += 1

        missing_episodes = {}
        for object_category, val in object_category_map.items():
            if val < 5:
                missing_episodes[object_category] = 5 - val
        missing_episode_data[key] = missing_episodes
        
        print("Missing episodes for scene: {} are: {}".format(key, len(list(missing_episodes))))
    write_json(missing_episode_data, "data/hit_data/objectnav_missing_episodes.json")


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
    parser.add_argument(
        "--max-episode-length", type=int, default=15000
    )
    parser.add_argument(
        "--append-dataset", dest='append_dataset', action='store_true'
    )
    parser.add_argument(
        "--thda", dest="is_thda", action="store_true"
    )
    parser.add_argument(
        "--sample", dest="sample", action="store_true"
    )
    parser.add_argument(
        "--gibson", dest="is_gibson", action="store_true"
    )
    args = parser.parse_args()
    replay_to_episode(
        args.replay_path, args.output_path, args.max_episodes,
        args.max_episode_length, append_dataset=args.append_dataset, is_thda=args.is_thda,
        sample=args.sample, is_gibson=args.is_gibson
    )
    list_missing_episodes()


if __name__ == '__main__':
    main()


