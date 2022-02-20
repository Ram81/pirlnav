import argparse
import csv
import copy
import datetime
import glob
import gzip
import json
import os
import re
import sys
import zipfile

from collections import defaultdict
from tqdm import tqdm
from habitat.datasets.utils import VocabFromText
from psiturk_dataset.utils.utils import load_dataset


max_instruction_len = 9
instruction_list = []
unique_action_combo_map = {}
max_num_actions = 0
num_actions_lte_tenk = 0
total_episodes = 0
excluded_ep = 0
task_episode_map = defaultdict(list)

filter_episodes = ['A3O5RKGH6VB19C:33LKR6A5KGNA8WOCNXOVC66E9LJ1TR', 'A2Q6L9LKSNU7EB:3WOKGM4L73JUOFYMVXYW4RHH5IL0OK', 'A3T5UAYFKJBVTA:3KIBXJ1WD7XW155QZQ8ENBWQJIAOK1', 'A3T5UAYFKJBVTA:3U5NZHP4LT5NKFGJ85IWZJLO157HPA', 'AOMFEAWQHU3D8:36W0OB37HYHHYJIPVEGYQHN2DFGZH3', 'A3VBNWON5XOUVS:3A1COHJ8NLY2ENH2MOGLDXQYCV28HC', 'A2TUUIV61CR0C7:3VNL7UK1XHM1YBIKUW3G18A8ASKTFW', 'A3T5UAYFKJBVTA:37TRT2X24SUH7RAZD03GGCWEXTWJBW', 'A1R0689JPSQ3OF:369J354OFFD1AD3393158JI6RZ9G67', 'A3T5UAYFKJBVTA:3Q5ZZ9ZEVQIQYUX7LMFCO0N87MG58X', 'A3T5UAYFKJBVTA:3EQHHY4HQUV5R93P4KR0GD46V5W5GG', 'A2CWA5VQZ6IWMQ:3HMVI3QICLV6PIN6X3BUKNYTOKPY1V', 'A3T5UAYFKJBVTA:33NF62TLXL5I0UETJDG9FFF4KN4JKF', 'A3T5UAYFKJBVTA:382M9COHEJIOSAXDZQ9KOMBAH67EUN', 'APGX2WZ59OWDN:3GNA64GUZG7W4YX37GWCAR881RDQ52', 'A1NSHNH3MNFRGW:3JAOYWH7VK74EOJ2I11X5LEGF4PL9R', 'AOMFEAWQHU3D8:3QRYMNZ7F0KDM3V1SKMT9DJHE5CNTD', 'A2TUUIV61CR0C7:3AZHRG4CU6N52Q50CZE4GOJ6ARP30S', 'A2TUUIV61CR0C7:3KOPY89HMA5C4W6MY7OAYTY290V3JZ', 'A1R0689JPSQ3OF:31N2WW6R9TTWZUKQFHXPJV6AGVRF3P', 'A1R0689JPSQ3OF:3Y4W8Q93L1NJDJ8D8L85EQV8V2HDVM', 'A14O8IA2LCQBDA:3Q8GYXHFER5SAXNK2YVHQMJRXCD5CD', 'A2Q6L9LKSNU7EB:317HQ483I9VNDPFQY8NPV6H812WNIM', 'A272X64FOZFYLB:3M1CVSFP628TA49K2CNTI2OUT5EQA8', 'A2TUUIV61CR0C7:3U5NZHP4LT5NKFGJ85IWZJLO15QHPT', 'AKYXQY5IP7S0Z:3ZDAD0O1T3GIYX95UQ927FAFGJ7TXO', 'A2TUUIV61CR0C7:336KAV9KYSVDE352G7B8P68YGQ3Y2C', 'A1R0689JPSQ3OF:3634BBTX0QXBPX290K4CN18ZZB4IF4', 'A2TUUIV61CR0C7:3MHW492WW2GMHDEQLE78XGI24FNMVP', 'A2CWA5VQZ6IWMQ:32SVAV9L3HC1333I41BX5UVJJD13AJ', 'A1R0689JPSQ3OF:3L4D84MIL1VRY4DLDSDC2NZCS74JHH', 'A3T5UAYFKJBVTA:3X1FV8S5JZUMP3I4AB9DKBY58TUGVU', 'A3L8LSM7V7KX3T:3VP0C6EFSIZ12NZPK6Z0LO23P2X6MH', 'A2Q6L9LKSNU7EB:31JLPPHS2WXQ57XJEKGF6PFO8U53OI', 'A2TUUIV61CR0C7:3JC6VJ2SADM4HIQMIKZQKRT3X9SA5G', 'A1R0689JPSQ3OF:3VA45EW49PQUV4J4RG2WIW0RAAKO14', 'A1R0689JPSQ3OF:3Z7ISHFUH2YO58HWSAMSD4U38MIZ8U', 'A3T5UAYFKJBVTA:3TAYZSBPLNBGIHTTH1JJ7KKXZ6W2ST', 'A3T5UAYFKJBVTA:3H7XDTSHKEUZ4SI90LE96DHJF3RGW1', 'A2TUUIV61CR0C7:3SPJ0334236DKZ3ANSH0ONUFI2XJYU', 'A2TUUIV61CR0C7:3LBXNTKX0TYZEI0RWK4LGF93KQK9XA', 'A1L3937MY09J3I:3D4CH1LGECWOSW517A4HST98H0HG9I', 'A2L26DMSVUEDP6:3IRIK4HM3CNOT1NY7H5MISXRVNB6CF', 'A2TUUIV61CR0C7:3KRVW3HTZPO6PLXMRJ23MTYV5AOSMJ', 'A2Q6L9LKSNU7EB:37UQDCYH6ZY3WA73H85JEYLC9PO7VT', 'A2TUUIV61CR0C7:3PXX5PX6LZ166Y7VJUQ3NDTV75EABH', 'A3T5UAYFKJBVTA:3R5F3LQFV4NRQ04CZRBOAQK3NQQZOZ', 'AOMFEAWQHU3D8:33LKR6A5KGNA8WOCNXOVC66EBU71TZ', 'A3T5UAYFKJBVTA:3S96KQ6I9O740R4O3Q8QD87NYHRTDC', 'A1R0689JPSQ3OF:3RSDURM96CP59JHI9R69R7HNDTNYE8', 'A3T5UAYFKJBVTA:3EKVH9QME07AGSABKBOUCLYXY3M2DZ', 'APGX2WZ59OWDN:3XXU1SWE8OY5MB4LLETE3WXCG1HA0V', 'A2TUUIV61CR0C7:37UQDCYH6ZY3WA73H85JEYLC9QVV7Q', 'A1ZE52NWZPN85P:3PWWM24LHU1YZXEK33DEQTKWPO9283', 'A2TUUIV61CR0C7:3T3IWE1XG8QYP08T8CEAD7EMHUTQTN', 'A1R0689JPSQ3OF:3B3WTRP3DD5YD2XU8VJSQPF77JX92H', 'A2TUUIV61CR0C7:3U088ZLJVMW2TO7OMJP6LLU38KDW0L', 'A3C8NUIBNZYMT2:36W0OB37HYHHYJIPVEGYQHN2B3QHZ5', 'ANLHIS7CZAZVQ:3LWJHTCVCEPO6VQSDS9LW3ZLQMBQF7', 'A3T5UAYFKJBVTA:3LOZAJ85YFGOEYFSBBP66S1PELXX2V', 'A2TUUIV61CR0C7:35USIKEBNTJ7K5KPW7E0Y3R3SCL6NX', 'A2TUUIV61CR0C7:3CN4LGXD5ZRNHHKPKLUWIL5W04ZY48', 'AIOOOO5OXWXKM:3SITXWYCNXCI2BFOU4IH7L4TAJNBXA', 'A1R0689JPSQ3OF:3GNCZX450KQ8AS852Z84IXYKOB8PA0', 'A1R0689JPSQ3OF:37UQDCYH6ZY3WA73H85JEYLC9R0V7X', 'A2TUUIV61CR0C7:34S9DKFK75S93PUV2Q9SHUBWRLONYM', 'A2TUUIV61CR0C7:3PWWM24LHU1YZXEK33DEQTKWNB1829', 'ADEMAGRRRGSPT:3DIP6YHAPEVQUDQ0WN8FSUTLKFD8EZ', 'A3T5UAYFKJBVTA:3LEIZ60CDL2OJD06X2S6D0PESR6Z9N', 'A3T5UAYFKJBVTA:3P4RDNWND79RUZO5JAVX2Z0RSJFIJ3', 'A2TUUIV61CR0C7:3U8YCDAGXRJX9RB2AAQ0TWCHKJ2Q0H', 'A2TUUIV61CR0C7:3TOK3KHVJVL86QY6GWJ5J6R4DWKO71', 'A1R0689JPSQ3OF:31T4R4OBOUJ7X113QRAEO6XNOKNC7I', 'A3T5UAYFKJBVTA:3JNQLM5FT6PTE4Y3XSMIVY627CNL2X', 'A2TUUIV61CR0C7:3R8YZBNQ9JLBR2BMV9B98BM4SX8Q7H', 'A2TUUIV61CR0C7:3X65QVEQI2Q6CMQ5ULBO7BFOJ26LCA']
scene_map = {
    "empty_house.glb": "29hnd4uzFmX.glb",
    "house.glb": "q9vSo1VnCiC.glb",
    "big_house.glb": "i5noydFURQK.glb",
    "big_house_2.glb": "S9hNv5qa7GM.glb",
    "bigger_house.glb": "JeFG25nYj2p.glb",
    "house_4.glb": "zsNo4HB9uLZ.glb",
    "house_5.glb": "TbHJrupSAjP.glb",
    "house_6.glb": "JmbYfDe2QKZ.glb",
    "house_8.glb": "jtcxE69GiFV.glb",
}


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
    if replay_path.endswith("zip"):
        archive = zipfile.ZipFile(replay_path, "r")
        for file_path in tqdm(archive.namelist()):
            reader = read_csv_from_zip(archive, file_path)

            episode, counts = convert_to_episode(reader)
            # Filter out episodes that have unstable initialization
            if episode["episode_id"] in filter_episodes:
                continue
            if len(episode["reference_replay"]) <= max_episode_length:
                episodes.append(episode)
                episode_lengths.append(counts)
            if sample:
                if len(episodes) >= max_episodes:
                    break
    else:
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

            # Filter out episodes that have unstable initialization
            if episode["episode_id"] in filter_episodes:
                continue
            if len(episode["reference_replay"]) <= max_episode_length:
                scene_episode_map[episode["scene_id"]].append(episode)
                all_episodes["episodes"].append(episode)
                episode_lengths.append(counts)
            if sample:
                if len(episode_lengths) >= max_episodes:
                    break
    print("Total duplicate episodes: {}".format(duplicates))

    objectnav_dataset_path = "data/datasets/objectnav_mp3d/objectnav_mp3d_thda_40k/train/content/{}.json.gz"
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


