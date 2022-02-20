import argparse
import copy
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gzip

from collections import defaultdict
from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from habitat.datasets.utils import VocabFromText
from psiturk_dataset.utils.utils import load_vocab

# eval success episodes
# episode_ids = ['A1L3937MY09J3I:3Z7EFSHGNBH1CG7U84ECI5ABGYYCX5','A1ZE52NWZPN85P:3C6FJU71TSWMYFE4ZRLEVP3QQU9YUY','A2CWA5VQZ6IWMQ:3YGXWBAF72KAEEJKOTC7LUDDNIP4C8','APGX2WZ59OWDN:358010RM5GWXBPDUZL9H8XY015IVXR']
# train success episodes
# episode_ids = ['A1NKBXOTZAI1YK:3SEPORI8WP22OWABP8669V0YPHWAZS','A1ZE52NWZPN85P:39OWYR0EPMUXFXHE42QF9P2NG19YF2','A272X64FOZFYLB:30X31N5D65T5NKOXUGCYD23V2FUSAT','A2CWA5VQZ6IWMQ:3OS46CRSLH2KSATYYY0R8KLG5LU6VO','A2Q6L9LKSNU7EB:323Q6SJS8KJBT2RPU2MRNP7KQ5NHFD','A2Q6L9LKSNU7EB:39U1BHVTDNU6IZ2RA12E0ZLBY95T3I']
# resnet1 eps
instruction_list = []
episode_ids = ['AKYXQY5IP7S0Z:3I0BTBYZAZO6IT2O1K7U6IFJAEIY0Q','A1NSHNH3MNFRGW:3GLB5JMZFZY0VMIIJQ9JEPSYZ2JGDB','A2TUUIV61CR0C7:3HYA4D452TM7ECO7BHJK0L1I02B2FH','AEWGY34WUIA32:3ZY8KE4ISL6D2SCID7EPEP274HEQV2','A1NSHNH3MNFRGW:3FTF2T8WLTLKPIV1MF8ZEWVW2V4W9Z','A272X64FOZFYLB:36W0OB37HYHHYJIPVEGYQHN22FPZH1','A1ZE52NWZPN85P:3VZLGYJEYNDEK9I40IYKT3BWQ0SZX4','ANBWJZYU2A68T:3LKC68YZ3C6NW5Z7O4RHBMQLWPTOWX','ADEMAGRRRGSPT:39JEC7537W498R2Z8PDUUKDQZAWVC0','AKYXQY5IP7S0Z:3ZQIG0FLQGJIMP84PGDV6EKTTXIWVU','A272X64FOZFYLB:3R3YRB5GRH6L2XG1JL7YS3LJNX7AUP','AKYXQY5IP7S0Z:3YJ6NA41JDJJBLB9W5LHBW13593PJU','A1NSHNH3MNFRGW:32M8BPYGAVPH3XY4B4AU5M8BRXWGIY','A272X64FOZFYLB:3X73LLYYQ3HNHU46SQ54VUGTSW9HNK','A182N7RLXGSCZG:3KGTPGBS6ZOWXULX66EJML2LAPC2U2','A272X64FOZFYLB:3VA45EW49PQUV4J4RG2WIW0R1HN1OP','A3O5RKGH6VB19C:320DUZ38G9PDY8IATMVUHNNB2FEGJ7','A2CWA5VQZ6IWMQ:3NLZY2D53RSA6N0OZ3CJRG45EPLLQ0','A2TUUIV61CR0C7:33TIN5LC06DOENQ11GQNZTGCDJHY99','AKYXQY5IP7S0Z:3Z4XG4ZF4AUZ0DHHRSY7GJESR1HX8D','A1ZE52NWZPN85P:32ZKVD547HQ6MD8AAFBT05FPS4GB3P','AEWGY34WUIA32:34S9DKFK75S93PUV2Q9SHUBWHP4YNB','A1ZE52NWZPN85P:36PW28KO41Z4D1JFTLSTOLZG1P4EAZ','A3O5RKGH6VB19C:3MMN5BL1W17254C71412ELQJ4663MY','A272X64FOZFYLB:3WS1NTTKE0F0I2LTWUF6HX834OLF09','AKYXQY5IP7S0Z:3483FV8BEGMBVJVWAOGG6FO573B26H','A3O5RKGH6VB19C:3EKVH9QME07AGSABKBOUCLYXOSK2D1','A1NKBXOTZAI1YK:3NG53N1RLXMUR4FQ51OQM6SPP8ZP8K','A272X64FOZFYLB:3TYCR1GOTEMJKF1FMZVWI9G9JRWLZI','A272X64FOZFYLB:324G5B4FB5BN396NEBHUT5VM60X07L','A2CWA5VQZ6IWMQ:3RGU30DZTCBDQIEW4PTPUS780Y3MJU','ADEMAGRRRGSPT:3WEV0KO0OOV3LRR9EQ3033B1NDESD8','ADEMAGRRRGSPT:3KRVW3HTZPO6PLXMRJ23MTYVVCPMS8','A2TUUIV61CR0C7:3IAEQB9FMGNWS88IYVD10SEMTWWWDP','A2CWA5VQZ6IWMQ:3EF8EXOTT3YGUTS7B3ARA0J52TKJ17','A272X64FOZFYLB:3L4D84MIL1VRY4DLDSDC2NZCJC4HJG','AOMFEAWQHU3D8:32XVDSJFP10DKMGOX4NXVBLRYXF2M5','APGX2WZ59OWDN:39JEC7537W498R2Z8PDUUKDQ02SCVY','A1L3VZLK8DCFNM:31IBVUNM9U2GB3M9ZR3V2QYTWO1VFQ','A2TUUIV61CR0C7:3AQF3RZ55ALVWD78YJVNQYIUH166FK','A2DDPSXH2X96RF:37C0GNLMHH6YYTTC7D0X2YF95OF6DY','AEWGY34WUIA32:32XVDSJFP10DKMGOX4NXVBLRXJU2MR','A2Q6L9LKSNU7EB:3BEFOD78W8WNN0VB1I6LOQIPIB9M4J','A2CWA5VQZ6IWMQ:333U7HK6IBIAMO8JRWUMB2KERKHJDF','AEWGY34WUIA32:3UN61F00HYSWGZC3KVLCFHIDOU4R5I','AOMFEAWQHU3D8:34PGFRQONQE9VU8A8RZC3Q9ZY33WJN','A3O5RKGH6VB19C:3FIUS151DX5376S9LGARKAVVANYGG4','A3KC26Z78FBOJT:369J354OFFD1AD3393158JI6IGVG6I','A1ZE52NWZPN85P:3D8YOU6S9GNKFV4YT8QMCYJXRGM6U4','A1NSHNH3MNFRGW:3FPRZHYEP0ALVR6GFW2T1H9WUGUV3G','A1NSHNH3MNFRGW:33LK57MYLV86OSW568SXUVU4BLXSZW','A1NSHNH3MNFRGW:386PBUZZXH0TK0WB4DSAUFSJ034LJ2','A1NSHNH3MNFRGW:32ZKVD547HQ6MD8AAFBT05FPS2CB3H','A2TUUIV61CR0C7:3R2PKQ87NYBHV7UQM78PIRS8NTDIMX','A1ZE52NWZPN85P:3KKG4CDWKK18GGCHC92GJ4C5I8T49L','A3TYITGZ7LBO54:3EG49X351WFCWZYTYD19W5I1LWW6XL','AOMFEAWQHU3D8:3TK8OJTYM3OS2GB3DUZ0EKCX00IVP1','A272X64FOZFYLB:358010RM5GWXBPDUZL9H8XY02P0XVG','A1ZE52NWZPN85P:38JBBYETQQDPBC3YKKI2BIDG91M4E0','A1NSHNH3MNFRGW:3PJUZCGDJ8J9ZHZJOCST0GSAKUI98Q','A3O5RKGH6VB19C:37XITHEISYCHFKLIZ58KTNONFKMCRO','A1NSHNH3MNFRGW:34MAJL3QP6QM1EN1V016SR9JIAQ43Z','A1NSHNH3MNFRGW:3V5Q80FXIZUCY08ERMIIZCCLYWY235','A2Q6L9LKSNU7EB:3OB0CAO74JSHTT8KZSEFCAE0WQ9YH7','A3O5RKGH6VB19C:3RUIQRXJBDRZFQKB7Y4NAU5B2L4LLJ','A2TUUIV61CR0C7:3MHW492WW2GMHDEQLE78XGI2VV0VMY','A2CWA5VQZ6IWMQ:3C6FJU71TSWMYFE4ZRLEVP3QQU4UYP','APGX2WZ59OWDN:39OWYR0EPMUXFXHE42QF9P2NFWAYFS','A272X64FOZFYLB:33JKGHPFYEX9985HJNLHNZOP9I3NMU','A3K1P4SPR8XS7R:3P1L2B7AD3S7LBN8KQKF2B95XQOLOF']
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

def load_duplicate_episodes(path="data/hit_data/duplicate_episode.json"):
    f = open(path, "r")
    data = json.loads(f.read())
    episode_ids = [d["episode_id"] for d in data]
    return episode_ids


def sample_episodes(path, output_path, per_scene_limit=10):
    data = load_dataset(path)

    print("Number of episodes {}".format(len(data["episodes"])))

    sample_episodes = {}
    sample_episodes["instruction_vocab"] = data["instruction_vocab"]
    sample_episodes["episodes"] = []
    scene_map = {}
    for episode in data["episodes"]:
        scene_id = episode["scene_id"]

        if scene_id not in scene_map.keys():
            scene_map[scene_id] = 0
        scene_map[scene_id] += 1 

        if scene_map[scene_id] <= per_scene_limit:
            sample_episodes["episodes"].append(episode)
    
    print("Sampled episodes: {}".format(len(sample_episodes["episodes"])))
    
    write_json(sample_episodes, output_path)
    write_gzip(output_path, output_path)


def sample_episodes_by_episode_ids(path, output_path):
    episode_file = open(path, "r")
    data = json.loads(episode_file.read())

    episode_ids = load_duplicate_episodes()

    print("Number of episodes {}".format(len(data["episodes"])))
    print("Number of duplicate episodes {}".format(len(episode_ids)))
    print("Sampling {} episodes".format(len(episode_ids)))

    sample_episodes = {}
    sample_episodes["instruction_vocab"] = data["instruction_vocab"]
    sample_episodes["episodes"] = []
    scene_map = {}
    for episode in data["episodes"]:
        scene_id = episode["scene_id"]
        episode_id = episode["episode_id"]
        # Exclude episodes
        if episode_id not in episode_ids:
            sample_episodes["episodes"].append(episode)
    
    print("Sampled episodes: {}".format(len(sample_episodes["episodes"])))
    
    write_json(sample_episodes, output_path)
    write_gzip(output_path, output_path)


def sample_objectnav_episodes(path, output_path, prev_tasks=[]):
    prev_tasks = [] # "data/datasets/objectnav_mp3d_v2/train/sampled", "data/datasets/objectnav_mp3d_v2/train/sampled_v2"]
    prev_episode_points = {}
    for prev_path in prev_tasks:
        prev_task_files = glob.glob(prev_path + "/*.json")
        for prev_task in prev_task_files:
            data = load_json_dataset(prev_task)
            for ep in data["episodes"]:
                key = str(ep["start_position"])
                if key not in prev_episode_points.keys():
                    prev_episode_points[key] = 0
                prev_episode_points[key] += 1

    files = glob.glob(path + "/*.json.gz")
    hits = []
    scene_ep_map = defaultdict(int)
    num_duplicates = 0
    total_episodes = 0
    print("Number of existing episodes: {}".format(len(prev_episode_points.keys())))
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        object_category_map = defaultdict(int)
        episodes = []
        count = 0
        for episode in data["episodes"]:
            key = str(episode["start_position"])
            if prev_episode_points.get(key) is not None and prev_episode_points[key] > 0:
                num_duplicates += 1
                continue
            object_category = episode["object_category"]
            if object_category_map[object_category] < 3*45:
                object_category_map[object_category] += 1
                episodes.append(episode)
                if key not in prev_episode_points.keys():
                    prev_episode_points[key] = 0
                prev_episode_points[key] += 1
                count += 1

        data["episodes"] = episodes
        dest_path = os.path.join(output_path, "{}.json".format(scene_id))
        # print(output_path)
        # print(dest_path)
        write_json(data, dest_path)
        write_gzip(dest_path, dest_path)

        scene_ep_map[scene_id] = len(data['episodes'])

        # data["episodes"] = episodes[:1]
        # dest_path = os.path.join(output_path, "{}_train.json".format(scene_id))
        # write_json(data, dest_path)
        total_episodes += len(episodes)

        ep = {
            "name": "{}.json".format(scene_id),
            "config": "tasks/objectnav_v2/{}.json".format(scene_id),
            "scene": "{}.glb".format(scene_id),
            "trainingTask": {
                "name": "{}_train.json".format(scene_id),
                "config": "tasks/objectnav_v2/{}_train.json".format(scene_id)
            }
        }

        hits.append(ep)
    # print(json.dumps(hits, indent=4))
    with open("hits.json", "w") as f:
        f.write(json.dumps(hits, indent=4))
    with open("scene_ep_map.json", "w") as f:
        f.write(json.dumps(scene_ep_map))

    print("Number of new episodes: {}".format(total_episodes))
    print("Number of duplicate episodes: {}".format(num_duplicates))
    print("Number of new episodes: {}".format(len(prev_episode_points.keys())))


def sample_episodes_by_scene(path, output_path, limit=500):
    data = load_dataset(path)

    ep_inst_map = {}
    episodes = []
    excluded_episodes = []
    for ep in tqdm(data["episodes"]):
        instruction = ep["instruction"]["instruction_text"].replace(" ", "_")
        scene_id = ep["scene_id"]

        if scene_id not in ep_inst_map.keys():
            ep_inst_map[scene_id] = {}
        else:
            if instruction not in ep_inst_map[scene_id].keys():
                ep_inst_map[scene_id][instruction] = 1
                episodes.append(ep)
            else:
                ep_inst_map[scene_id][instruction] += 1
                excluded_episodes.append(ep)

    sample_length = limit - len(episodes)
    if sample_length > 0:
        sampled = np.random.choice(excluded_episodes, sample_length)
        episodes.extend(sampled.tolist())
    else:
        sampled = np.random.choice(episodes, limit)
        episodes = sampled.tolist()

    data["episodes"] = episodes

    write_json(data, output_path)
    write_gzip(output_path, output_path)


def sample_objectnav_episodes_custom(path, output_path, episode_list_json_path):
    files = glob.glob(path + "/*.json.gz")
    print("In ObjectNav episode sampler")
    # episode_ids = ["A2DDPSXH2X96RF:3DPNQGW4LNILYXAJE2Z4ZUL33AU642", "A7XL1V3G7C2VV:3QY5DC2MXTNGYOX9U1TQ64WAVJ6UFV", "A272X64FOZFYLB:3SNLUL3WO6Q2YG75GCWO1H1UQ67LUM", "AEWGY34WUIA32:32UTUBMZ7IZQYMATUPHZJ078SJ2BVE", "AOMFEAWQHU3D8:34S9DKFK75S93PUV2Q9SHUBWT7RYNA", "AOMFEAWQHU3D8:3MRNMEIQW79GHEWJUH6ZRHX66OODLI", "A1ZE52NWZPN85P:3TOK3KHVJVL86QY6GWJ5J6R4F8I7O8", "A3KC26Z78FBOJT:3NC5L260MQPLLJDCYFHH7Y4LD55FO6"]
    # train episodes
    f = open(episode_list_json_path)
    episode_ids = json.loads(f.read())
    print("Total episodes to sample: {}".format(len(episode_ids)))
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        episodes = []
        for episode in data["episodes"]:
            if episode["episode_id"] in episode_ids:
                episodes.append(episode)
        data["episodes"] = episodes
        if len(data["episodes"]) > 0:
            dest_path = os.path.join(output_path, "{}.json".format(scene_id))
            print("Writing at: {} - {} episodes".format(dest_path, len(episodes)))
            write_json(data, dest_path)
            write_gzip(dest_path, dest_path)


def sample_objectnav_episodes_visualization(path, output_path):
    files = glob.glob(path + "/*.json.gz")
    print("In ObjectNav vis episode sampler")
    # train episodes
    scene_ids = ['ULsKaCPVFJR.glb']
    min_ep_length = 100
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        episodes = []
        for episode in data["episodes"]:
            if episode["object_category"] != "stool":
                continue
            if episode["scene_id"].split("/")[-1] in scene_ids and len(episode["reference_replay"]) > min_ep_length:
                episodes.append(episode)
        data["episodes"] = episodes
        if len(data["episodes"]) > 0 and episode["scene_id"].split("/")[-1] in scene_ids:
            dest_path = os.path.join(output_path, "{}.json".format(scene_id))
            print("Wrting at: {}".format(dest_path))
            write_json(data, dest_path)
            write_gzip(dest_path, dest_path)
        if episode["scene_id"].split("/")[-1] in  scene_ids:
            print("Num episodes: {} for scene: {}".format(len(episodes), episode["scene_id"]))


def sample_coverage_episodes(path, output_path, total_episodes=1200, task="objectnav"):
    files = glob.glob(path + "/*.json.gz")
    print("In coverage episode sampler")
    # train episodes
    scene_ids = ["ULsKaCPVFJR"]
    ep_per_scene = 20
    filtered_episodes = 0
    ep_scene_map = defaultdict(list)
    if path.split(".")[-1] == "gz":
        files = [path]
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        episodes = []
        if scene_id not in scene_ids and len(scene_ids) > 0:
            continue
        for episode in tqdm(data["episodes"]):
            s_id = episode["scene_id"]
            if episode["object_category"] != "stool":
                continue
            if len(ep_scene_map[s_id]) >= ep_per_scene:
                continue
            ep_scene_map[s_id].append(1)
            episodes.append(episode)
            filtered_episodes += 1

            if filtered_episodes >= total_episodes:
                break
        data["episodes"] = episodes
        if len(data["episodes"]) > 0:
            dest_path = os.path.join(output_path, "{}.json".format(scene_id))
            if output_path.split(".")[-1] == "json":
                dest_path = output_path
            
            print("Wrting at: {}".format(dest_path))
            # write_json(data, dest_path)
            # write_gzip(dest_path, dest_path)
        print("Total episodess: {}".format(len(data["episodes"])))


def sample_stratified_objectnav_dataset(path, output_path, total_episodes=1200, task="objectnav"):
    files = glob.glob(path + "/*.json.gz")
    print("In stratified episode sampler")
    # train episodes
    df = []
    filtered_episodes = 0
    ep_scene_map = defaultdict(list)
    if path.split(".")[-1] == "gz":
        files = [path]
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        for episode in tqdm(data["episodes"]):
            scene_goal = "{}_{}".format(episode["scene_id"], episode["object_category"])
            df.append(([episode["episode_id"], scene_goal]))

    df = pd.DataFrame(df, columns=['episode_id', 'goal'])
    test_size = total_episodes / df.shape[0]
    s_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

    train_splits, val_splits = [], []
    for train_split, val_split in s_split.split(df['episode_id'], df['goal']):
        train_splits.append(train_split)
        val_splits.append(val_split)
    
    indices = val_splits[0]

    filtered_episode_ids = df['episode_id'][indices].values
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        episodes = []
        for episode in tqdm(data["episodes"]):
            if episode["episode_id"] in filtered_episode_ids:
                episodes.append(episode)
                filtered_episodes += 1

        data["episodes"] = episodes
        if len(data["episodes"]) > 0:
            dest_path = os.path.join(output_path, "{}.json".format(scene_id))
            if output_path.split(".")[-1] == "json":
                dest_path = output_path
            
            print("Wrting at: {}".format(dest_path))
            write_json(data, dest_path)
            write_gzip(dest_path, dest_path)
        print("Total episodess: {}".format(len(data["episodes"])))
    print("Sampled dataset size: {}".format(len(filtered_episode_ids)))


def append_instruction(instruction):
    instruction_list.append(instruction)


def compute_instruction_tokens(episodes):
    vocab = load_vocab()
    instruction_vocab = VocabFromText(
        sentences=vocab["sentences"]
    )
    max_token_size = 0
    for episode in episodes:
        instruction = episode["instruction"]["instruction_text"]
        instruction_tokens = instruction_vocab.tokenize_and_index(instruction, keep=())
        max_token_size = max(max_token_size, len(instruction_tokens))

    for episode in episodes:
        instruction = episode["instruction"]["instruction_text"]
        instruction_tokens = instruction_vocab.tokenize_and_index(instruction, keep=())
        if len(instruction_tokens) < max_token_size:
            instruction_tokens = instruction_tokens + [instruction_vocab.word2idx("<pad>")] * (max_token_size - len(instruction_tokens))
        episode["instruction"]["instruction_tokens"] = instruction_tokens
    return episodes


def populate_episodes_points(episode, visited_point_dict):
    is_duplicate = False
    point = str(episode["start_position"])
    if visited_point_dict.get(point):
        visited_point_dict[point] += 1
        # print("Redundant agent position in episode {}".format(episode["episode_id"]))
        is_duplicate = True
    else:
        visited_point_dict[point] = 1

    for object_ in episode["objects"]:
        point = str(object_["position"])
        if visited_point_dict.get(point):
            visited_point_dict[point] += 1
            is_duplicate = True
        else:
            visited_point_dict[point] = 1
    return is_duplicate

def merge_object_rearrangement_episodes(path, output_path):
    files = glob.glob(path)
    base_file = "data/datasets/object_rearrangement/s_path_v4/train/train.json.gz"
    dataset = load_dataset(base_file)

    all_episodes = {
        "episodes": []
    }
    print("In merge rearrangement episode")
    # files = ["data/tasks/house_v3.json"]
    visited_point_dict = {}
    duplicates = 0
    total_episodes = 0
    for file in files:
        dset = load_json_dataset(file)

        ep_id = 0
        episodes = []
        for data in dset["episodes"]:
            episode = copy.deepcopy(data)
            episode["episode_id"] = ep_id
            episode["scene_id"] = scene_map[data["scene_id"]]
            episode["start_position"] = data["start_position"]
            episode["start_rotation"] = data["start_rotation"]

            episode["objects"] = []
            for idx in range(len(data["objects"])):
                object_data = {}
                object_data["object_id"] = data["objects"][idx]["objectId"]
                object_data["object_template"] = data["objects"][idx]["objectHandle"]
                object_data["position"] = data["objects"][idx]["position"]
                object_data["rotation"] = data["objects"][idx]["rotation"]
                object_data["motion_type"] = data["objects"][idx]["motionType"]
                object_data["is_receptacle"] = data["objects"][idx]["isReceptacle"]
                episode["objects"].append(object_data)

            instruction_text = data["task"]["instruction"]
            if "place the shark" in instruction_text:
                instruction_text = instruction_text.replace("place the shark", "place the toy shark")
            episode["instruction"] = {
                "instruction_text": instruction_text.lower(),
            }
            append_instruction(instruction_text)
            if "goals" in data["task"].keys():
                goals = []
                for object_ in data["objects"]:
                    goals.append({
                        "position": object_["position"],
                        "rotation": object_["rotation"],
                        "info": {
                            "is_receptacle": object_["isReceptacle"],
                        }
                    })
                episode["goals"] = goals
            episode["reference_replay"] = []
            del episode["task"]

            is_duplicate = populate_episodes_points(episode, visited_point_dict)
            total_episodes += 1
            if is_duplicate:
                duplicates += 1 
                continue
            episodes.append(episode)
            ep_id += 1
        all_episodes["episodes"].extend(episodes)

    print("Total duplicate episodes: {}/{}".format(duplicates, total_episodes))
    vocab = load_vocab()
    compute_instruction_tokens(all_episodes["episodes"])
    all_episodes["instruction_vocab"] = {
        "sentences": vocab["sentences"]
    }
    write_json(all_episodes, output_path)
    write_gzip(output_path, output_path)


def check_duplicates(input_path_1, input_path_2):
    print("In check duplicates")
    ep_point_dict = {}
    total_episodes = 0
    duplicates = 0
    paths = [input_path_1, input_path_2]
    for path in paths:
        files = glob.glob(path)
        for f in files:
            d = load_dataset(f)
            for ep in d["episodes"]:
                key = str(ep["start_position"])
                if ep_point_dict.get(key) != 1:
                    ep_point_dict[key] = 1
                else:
                    duplicates += 1
                total_episodes += 1
    print("Total episodes: {}, Duplicates: {}".format(total_episodes, duplicates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_approvals/hits_max_length_1500.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sample_hits.json"
    )
    parser.add_argument(
        "--per-scene-limit", type=int, default=10
    )
    parser.add_argument(
        "--limit", type=int, default=10
    )
    parser.add_argument(
        "--per-scene", dest='per_scene', action='store_true'
    )
    parser.add_argument(
        "--sample-episodes", dest='sample_episodes', action='store_true'
    )
    parser.add_argument(
        "--objectnav", dest='is_objectnav', action='store_true'
    )
    parser.add_argument(
        "--prev-tasks", type=str, default="data/datasets/objectnav_mp3d_v2/train/sampled/"
    )
    parser.add_argument(
        "--total-episodes", type=int, default=10
    )
    parser.add_argument(
        "--ep-list", type=str, default="data/tasks/objectnav_train_split.json"
    )
    parser.add_argument(
        "--task", type=str, default="objectnav"
    )
    parser.add_argument(
        "--check-duplicates", dest="check_duplicates", action="store_true"
    )
    args = parser.parse_args()
    # if args.sample_episodes and not args.is_objectnav and not args.per_scene:
    #     sample_episodes_by_episode_ids(args.input_path, args.output_path)
    # elif args.sample_episodes and args.is_objectnav:
    #     sample_objectnav_episodes(args.input_path, args.output_path, args.prev_tasks)
    # elif args.per_scene and args.sample_episodes:
    #     sample_episodes_by_scene(args.input_path, args.output_path, args.limit)
    # else:
    #     sample_episodes(args.input_path, args.output_path, args.per_scene_limit)
    # sample_objectnav_episodes_custom(args.input_path, args.output_path)
    # sample_coverage_episodes(args.input_path, args.output_path)
    # sample_stratified_objectnav_dataset(args.input_path, args.output_path, args.total_episodes)
    if args.task == "rearrangement":
        merge_object_rearrangement_episodes(args.input_path, args.output_path)
    else:
        if args.check_duplicates:
            check_duplicates(args.input_path, args.output_path)
        else:
            sample_objectnav_episodes(args.input_path, args.output_path)