import argparse
import csv
import copy
import datetime
import glob
import gzip
import json
import os
import sys
import random
import zipfile

from collections import defaultdict
from io import TextIOWrapper
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

csv.field_size_limit(sys.maxsize)

scene_dataset_path = {
    "mp3d": "data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_v1/{}/content/{}.json.gz",
    "gibson": "data/datasets/objectnav/objectnav_gibson/objectnav_gibson_v2/{}/content/{}.json.gz",
    "hm3d": "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/{}/content/{}.json.gz",
    "mp3d_thda": "data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda/train/content/{}.json.gz",
}


def read_csv(path, delimiter=","):
    file = open(path, "r")
    reader = csv.reader(file, delimiter=delimiter)
    return reader


def read_csv_from_zip(archive, path, delimiter=","):
    file = archive.open(path, "r")
    reader = csv.reader(TextIOWrapper(file, "utf-8"), delimiter=delimiter)
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
    elif action == "pause":
        return "PAUSE"
    return "STOP"


def is_thda_episode(scene_dataset, episode):
    return (scene_dataset in ["mp3d_thda"] and episode.get("is_thda"))


def parse_replay_data_for_action(action, data):
    replay_data = {}
    replay_data["action"] = action
    if action == "grabReleaseObject":
        replay_data["is_grab_action"] = data["actionData"]["grabAction"]
        replay_data["is_release_action"] = data["actionData"]["releaseAction"]
        replay_data["object_under_cross_hair"] = data["actionData"]["objectUnderCrosshair"]
        replay_data["gripped_object_id"] = data["actionData"]["grippedObjectId"]

        action_data = {}

        if replay_data["is_release_action"]:
            action_data["new_object_translation"] = data["actionData"]["actionMeta"]["newObjectTranslation"]
            action_data["new_object_id"] = data["actionData"]["actionMeta"]["newObjectId"]
            action_data["object_handle"] = data["actionData"]["actionMeta"]["objectHandle"]
            action_data["gripped_object_id"] = data["actionData"]["actionMeta"]["grippedObjectId"]
        elif replay_data["is_grab_action"]:
            action_data["gripped_object_id"] = data["actionData"]["actionMeta"]["grippedObjectId"]

        replay_data["action_data"] = action_data
    else:
        replay_data["collision"] = data["collision"]
        replay_data["object_under_cross_hair"] = data["objectUnderCrosshair"]
        replay_data["nearest_object_id"] = data["nearestObjectId"]
        replay_data["gripped_object_id"] = data["grippedObjectId"]
    if "agentState" in data.keys():
        replay_data["agent_state"] = {
            "position": data["agentState"]["position"],
            "rotation": data["agentState"]["rotation"],
            "sensor_data": data["agentState"]["sensorData"]
        }
        replay_data["object_states"] = get_object_states(data)

    return replay_data



def handle_step(step, episode, unique_id, timestamp):
    tried_submitting = False
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
                "action": action,
                "agent_state": None
            }
            if data.get("agentState") is not None:
                replay_data["agent_state"] = {
                    "position": data["agentState"]["position"],
                    "rotation": data["agentState"]["rotation"],
                    "sensor_data": data["agentState"]["sensorData"]
                }
            episode["reference_replay"].append(replay_data)
        elif step["event"] == "handleValidation":
            tried_submitting = True


    elif step.get("type"):
        if step["type"] == "finishStep":
            return True, tried_submitting
    return False, tried_submitting


def convert_to_episode(csv_reader):
    episode = {}
    viewer_step = False
    start_ts = 0
    end_ts = 0
    total_stops = 0
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
            is_viewer_step_finished, tried_submitting = handle_step(data, episode, unique_id, timestamp)
            total_stops += int(tried_submitting)
        end_ts = int(timestamp)
    if len(episode["reference_replay"]) == 0:
        return None, {}

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
        "hit_duration": hit_duration,
        "total_stops": total_stops
    }
    return episode, episode_length


def replay_to_episode(
    path,
    output_path,
    append_dataset=False,
    scene_dataset="mp3d",
    split="train",
    sample=None,
):
    all_episodes = {
        "episodes": []
    }

    episode_lengths = []
    start_pos_map = {}
    episodes = []
    scene_episode_map = defaultdict(list)
    duplicates = 0
    episode_ids = []
    archive = zipfile.ZipFile(path, "r")
    files = archive.namelist()

    if sample is not None:
        files = random.sample(files, sample)
        print("Sampling: {} episodes".format(sample))

    for file_path in tqdm(files):
        if not file_path.endswith("csv"):
            continue
        reader = read_csv_from_zip(archive, file_path)

        episode, counts = convert_to_episode(reader)
        if episode is None:
            continue
        episode["attempts"] = counts["total_stops"]

        episode_key = str(episode["start_position"]) + "_{}".format(episode["scene_id"])
        episode_ids.append({
            "episodeId": episode["episode_id"]
        })

        if episode_key in start_pos_map.keys():
            duplicates += 1
            continue
        start_pos_map[episode_key] = 1

        if scene_dataset not in episode["scene_id"] or is_thda_episode(scene_dataset, episode):
            continue

        scene_episode_map[episode["scene_id"]].append(episode)
        all_episodes["episodes"].append(episode)
        episode_lengths.append(counts)
    print("Total duplicate episodes: {}".format(duplicates))

    objectnav_dataset_path = scene_dataset_path[scene_dataset]
    print(len(scene_episode_map.keys()))

    for scene, episodes in scene_episode_map.items():
        scene = scene.split("/")[-1].split(".")[0]
        if not os.path.isfile(objectnav_dataset_path.format(split, scene)):
            print("Source dataset missing: {}".format(scene))
            continue

        if scene_dataset == "hm3d":
            for episode in episodes:
                episode["scene_dataset_config"] = "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        episode_data = load_dataset(objectnav_dataset_path.format(split, scene))
        episode_data["episodes"] = episodes

        path = os.path.join(output_path, "{}.json".format(scene))
        print(path)

        if append_dataset:
            existing_episodes = load_dataset(path + ".gz")
            len_before = len(episode_data["episodes"])
            episode_data["episodes"].extend(existing_episodes["episodes"])
            print("Appending new episodes to existing scene: {} -- {} -- {}".format(scene, len_before, len(episode_data["episodes"])))

        write_json(episode_data, path)
        write_gzip(path, path)
    
    write_json(episode_ids, "{}/hit_meta.json".format(output_path))
    show_average(all_episodes, episode_lengths)


def show_average(all_episodes, episode_lengths):
    print("Total episodes: {}".format(len(all_episodes["episodes"])))

    total_episodes = len(all_episodes["episodes"])
    total_submit_tries = 0
    total_hit_duration = 0
    episode_submission_tries_gt_1 = 0

    total_actions = 0
    num_eps_gt_than_2k = 0
    for episode_length  in episode_lengths:
        total_hit_duration += episode_length["hit_duration"]
        total_actions += episode_length["actual_episode_length"]
        num_eps_gt_than_2k += 1 if episode_length["actual_episode_length"] > 1900 else 0
        total_submit_tries += episode_length["total_stops"]
        episode_submission_tries_gt_1 += int(episode_length["total_stops"] > 1)

    print("\n\n")
    print("Average hit duration")
    print("All hits: {}, Total duration: {}, Num episodes: {}".format(round(total_hit_duration / total_episodes, 2), total_hit_duration, total_episodes))
    
    print("\n\n")
    print("Average episode length:")
    print("All Hits: {}, Num actions: {}, Num episodes: {}".format(round(total_actions / total_episodes, 2), total_actions, total_episodes))

    print("\n\n")
    print("Average submission tries:")
    print("All Hits: {}, Num episodes: {}".format(round(total_submit_tries / total_episodes, 2), total_episodes))
    print("Retries greater than 1 Hits: {}, Num episodes: {}".format(round(episode_submission_tries_gt_1 / total_episodes, 2), total_episodes))

    print("\n\n")
    print("Episodes greater than 1.9k actions: {}".format(num_eps_gt_than_2k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_data"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/data.json"
    )
    parser.add_argument(
        "--append-dataset", dest='append_dataset', action='store_true'
    )
    parser.add_argument(
        "--scene-dataset", type=str, default="mp3d"
    )
    parser.add_argument(
        "--split", type=str, default="train"
    )
    parser.add_argument(
        "--sample", type=int, default=None
    )
    args = parser.parse_args()
    replay_to_episode(
        args.path,
        args.output_path,
        append_dataset=args.append_dataset,
        scene_dataset=args.scene_dataset,
        split=args.split,
        sample=args.sample,
    )


if __name__ == '__main__':
    main()
