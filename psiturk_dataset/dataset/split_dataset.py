import argparse
import glob
import gzip
import json
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import defaultdict
from psiturk_dataset.utils.utils import (
    load_dataset,
    load_json_dataset,
    load_vocab,
    write_json,
    write_gzip,
    get_episode_idx_by_instructions,
    get_episode_idx_by_scene_ids,
    get_episodes_by_episode_ids,
    get_episodes_by_episode_index,
    get_unique_scenes
)
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold


def check_overlap(train_episode_idx, eval_episode_idx):
    print("Train episode indices: {}, Unique indices {}".format(len(train_episode_idx), len(set(train_episode_idx))))
    print("Eval episode indices: {}, Unique indices {}".format(len(eval_episode_idx), len(set(eval_episode_idx))))

    print("Train episode indices: {}".format(train_episode_idx[:20]))
    print("Eval episode indices: {}".format(eval_episode_idx[:20]))

    print("Overlap indices: {}".format(len(set(train_episode_idx).intersection(set(eval_episode_idx)))))


def train_val_split_episodes(data):
    X = []
    y = []
    instruction_map = {}
    for episode in data["episodes"]:
        yy_id = "{}:{}".format(episode["scene_id"], episode["instruction"]["instruction_text"])
        X.append("{}".format(episode["episode_id"]))
        y.append(yy_id)
        if yy_id not in instruction_map.keys():
            instruction_map[yy_id] = 0
        instruction_map[yy_id] += 1

    single_instruction = []
    exclude_idx = []
    for key, value in instruction_map.items():
        if value == 1:
            single_instruction.append(key)
            exclude_idx.append(y.index(key))
    
    instructions = list(instruction_map.keys())
    
    labels = [instructions.index(i) for i in y]
    y = [i for i in range(len(y))]

    y_idxs = []
    labels_filtered = []
    print("Total episodes: {}, Labels: {}".format(len(y), len(labels)))
    for ii in range(len(y)):
        y_i = y[ii]
        label_i = labels[ii]
        if y_i not in exclude_idx:
            y_idxs.append(y_i)
            labels_filtered.append(label_i)
    
    print("Single instance of instructions: {}".format(len(single_instruction)))
    print("Total episodes: {}, Filtered: {}".format(len(y), len(y_idxs)))

    sk_fold = StratifiedKFold(n_splits=5, shuffle=True)
    train_idxs = []
    eval_idxs = []
    for train_episode_idx, eval_episode_idx in sk_fold.split(y_idxs, labels_filtered):
        train_episode_idx = [y_idxs[i] for i in train_episode_idx.tolist()]
        eval_episode_idx = [y_idxs[i] for i in eval_episode_idx.tolist()]
        # Add samples with only 1 demo after splitting
        train_episode_idx.extend(exclude_idx)

        check_overlap(train_episode_idx, eval_episode_idx)
        
        train_idxs.append(train_episode_idx)
        eval_idxs.append(eval_episode_idx)

    return train_idxs, eval_idxs


def train_val_split_instruction_holdout(data):
    vocab = load_vocab()
    instructions = vocab["sentences"]

    rs_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idxs = []
    eval_idxs = []
    for train_instructions, eval_instructions in rs_split.split(instructions):
        train_instructions = [instructions[idx] for idx in train_instructions]
        eval_instructions = [instructions[idx] for idx in eval_instructions]

        train_episode_idx = get_episode_idx_by_instructions(data["episodes"], train_instructions)
        eval_episode_idx = get_episode_idx_by_instructions(data["episodes"], eval_instructions)

        check_overlap(train_episode_idx, eval_episode_idx)

        train_idxs.append(train_episode_idx)
        eval_idxs.append(eval_episode_idx)

    return train_idxs, eval_idxs


def train_val_split_scene_holdout(data):
    vocab = load_vocab()
    # scenes = ["house.glb", "empty_house.glb", "bigger_house.glb", "big_house.glb", "big_house_2.glb"]
    scenes = get_unique_scenes(data["episodes"])
    # train_scenes = ["JeFG25nYj2p.glb", "q9vSo1VnCiC.glb", "S9hNv5qa7GM.glb", "zsNo4HB9uLZ.glb", "jtcxE69GiFV.glb", "TbHJrupSAjP.glb", "JmbYfDe2QKZ.glb"]
    # eval_scenes = ["i5noydFURQK.glb", "29hnd4uzFmX.glb"]

    rs_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idxs = []
    eval_idxs = []
    for train_scenes, eval_scenes in rs_split.split(scenes):
        train_scenes = [scenes[i] for i in train_scenes]
        eval_scenes = [scenes[i] for i in eval_scenes]
        print("Train scenes: {}".format(train_scenes))
        print("Eval scenes: {}".format(eval_scenes))

        train_episode_idx = get_episode_idx_by_scene_ids(data["episodes"], train_scenes)
        eval_episode_idx = get_episode_idx_by_scene_ids(data["episodes"], eval_scenes)

    check_overlap(train_episode_idx, eval_episode_idx)

    train_idxs.append(train_episode_idx)
    eval_idxs.append(eval_episode_idx)

    return train_idxs, eval_idxs


def save_splits(data, indices, path):
    split_data = {
        "episodes": [],
        "instruction_vocab": data["instruction_vocab"]
    }
    split_data["episodes"] = get_episodes_by_episode_index(data["episodes"], indices)

    write_json(split_data, path)
    write_gzip(path, path)


def get_episode_data(data, episode_ids):
    split_data = {
        "episodes": [],
        "instruction_vocab": data["instruction_vocab"]
    }
    instructions = []
    split_data["episodes"] = get_episodes_by_episode_ids(data["episodes"], episode_ids)
    return split_data


def get_episode_ids(data, indices):
    episode_ids = []
    for i in range(len(data["episodes"])):
        episode = data["episodes"][i]
        if i in indices:
            episode_ids.append(episode["episode_id"])
    
    return episode_ids


def get_episode_ids(data):
    episode_ids = []
    for i in range(len(data["episodes"])):
        episode = data["episodes"][i]
        episode_ids.append(episode["episode_id"])
    
    return episode_ids


def split_data(path, output_path, instruction_holdout=False, scene_holdout=False):
    data = load_dataset(path)
    train_idxs = []
    eval_idxs = []
    if instruction_holdout:
        train_idxs, eval_idxs = train_val_split_instruction_holdout(data)
    elif scene_holdout:
        train_idxs, eval_idxs = train_val_split_scene_holdout(data)
    else:
        train_idxs, eval_idxs = train_val_split_episodes(data)

    for i, (train_idx, eval_idx) in enumerate(zip(train_idxs, eval_idxs)):
        print("Split {}: Train: {}, Eval: {}".format(i, len(train_idx), len(eval_idx)))
        train_data_path = output_path + "train_{}.json".format(i)
        eval_data_path = output_path + "eval_{}.json".format(i)
        print(train_data_path)
        save_splits(data, train_idx, train_data_path)
        save_splits(data, eval_idx, eval_data_path)


def validate_split_data(path, train_data_path, eval_data_path):
    data = load_dataset(path)
    train_idx, eval_idx = train_val_split_episodes(data)
    train_episode_ids = get_episode_ids(data, train_idx)
    eval_episodes_ids = get_episode_ids(data, eval_idx)
    validate_existing_split_overlap(train_data_path, eval_data_path, train_episode_ids, eval_episodes_ids)


def create_split_from_existing_data(path, train_data_path, eval_data_path, output_path):
    data = load_dataset(path)
    train_data = load_dataset(train_data_path)
    eval_data = load_dataset(eval_data_path)

    train_episode_ids = get_episode_ids(train_data)
    eval_episodes_ids = get_episode_ids(eval_data)

    print("Length train ep: {}".format(len(train_episode_ids)))
    print("Length eval ep: {}".format(len(eval_episodes_ids)))

    train_episodes_new = get_episode_data(data, train_episode_ids)
    eval_episodes_new = get_episode_data(data, eval_episodes_ids)

    print("\n\nLength train eps: {}".format(len(train_episodes_new['episodes'])))
    print("Length eval eps: {}".format(len(eval_episodes_new['episodes'])))

    train_output_path = output_path + "train.json"
    eval_output_path = output_path + "eval.json"
    write_json(train_episodes_new, train_output_path)
    write_json(eval_episodes_new, eval_output_path)
    write_gzip(train_output_path, train_output_path)
    write_gzip(eval_output_path, eval_output_path)


def validate_data(path, train_data_path, eval_data_path):
    train_data = load_json_dataset(train_data_path)
    eval_data = load_json_dataset(eval_data_path)

    train_episode_ids = []
    train_instructions = []
    train_scene_map = {}
    for episode in train_data["episodes"]:
        train_episode_ids.append(episode["episode_id"])
        train_instructions.append(episode["instruction"]["instruction_text"])
        scene_id = episode["scene_id"]
        if scene_id not in train_scene_map.keys():
            train_scene_map[scene_id] = 0
        train_scene_map[scene_id] += 1

    eval_episode_ids = []
    eval_instructions = []
    eval_scene_map = {}
    for episode in eval_data["episodes"]:
        eval_episode_ids.append(episode["episode_id"])
        eval_instructions.append(episode["instruction"]["instruction_text"])
        scene_id = episode["scene_id"]
        if scene_id not in eval_scene_map.keys():
            eval_scene_map[scene_id] = 0
        eval_scene_map[scene_id] += 1

    print("\nOverlap episodes: {}".format(len(set(train_episode_ids).intersection(set(eval_episode_ids)))))
    print("Unique train episodes: {}".format(len(set(train_episode_ids))))
    print("Unique eval episodes: {}".format(len(set(eval_episode_ids))))

    print("\nOverlap instructions: {}".format(len(set(train_instructions).intersection(set(eval_instructions)))))
    print("Unique instructions train episodes: {}".format(len(set(train_instructions))))
    print("Unique instructions eval episodes: {}".format(len(set(eval_instructions))))

    print("\nTrain data scene map: {}".format(train_scene_map))
    print("Eval data scene map: {}".format(eval_scene_map))
    
    train_scene_data = pd.DataFrame(train_scene_map.items(), columns=["scene", "num_episodes"])
    eval_scene_data = pd.DataFrame(eval_scene_map.items(), columns=["scene", "num_episodes"])

    sns.barplot(y="scene", x="num_episodes", data=train_scene_data)
    plt.gca().set(title="Train data distribution")
    plt.savefig("analysis/train_data_distribution.jpg")
    plt.clf()

    sns.barplot(y="scene", x="num_episodes", data=eval_scene_data)
    plt.gca().set(title="Eval data distribution")
    plt.savefig("analysis/eval_data_distribution.jpg")
    plt.clf()


def validate_existing_split_overlap(train_data_path, eval_data_path, train_ep_ids, eval_ep_ids):
    train_data = load_dataset(train_data_path)
    eval_data = load_dataset(eval_data_path)

    train_episode_ids = []
    train_instructions = []
    train_scene_map = {}
    for episode in train_data["episodes"]:
        train_episode_ids.append(episode["episode_id"])

    eval_episode_ids = []
    eval_instructions = []
    eval_scene_map = {}
    for episode in eval_data["episodes"]:
        eval_episode_ids.append(episode["episode_id"])

    print("\nOverlap train episodes: {}".format(len(set(train_episode_ids).intersection(set(train_ep_ids)))))
    print("Unique train episodes existing: {}".format(len(set(train_episode_ids))))
    print("Unique train episodes new: {}".format(len(set(train_ep_ids))))

    print("\nOverlap eval episodes: {}".format(len(set(eval_episode_ids).intersection(set(eval_ep_ids)))))
    print("Unique eval episodes existing: {}".format(len(set(eval_episode_ids))))
    print("Unique eval episodes new: {}".format(len(set(eval_ep_ids))))


def convert_objectnav_episodes_to_scene_split(input_path, output_path):
    files = glob.glob(input_path + "/*json.gz")
    print("Reading files: {}".format(len(files)))
    scene_ep_map = defaultdict(list)
    total = 0
    for f in files:
        if "_failed" in f:
            continue
        d = load_dataset(f)
        for ep in d['episodes']:
            scene_id = ep['scene_id'].split("/")[-1].split(".")[0]
            ep['scene_id'] = ep['scene_id'].replace("data/scene_datasets/", "")
            scene_ep_map[scene_id].append(ep)
        total +=  len(d['episodes'])
    
    print("Total episodes: {}".format(total))
    objectnav_episodes = "data/datasets/objectnav_mp3d_v1/train/content"
    for scene_id, episodes in scene_ep_map.items():
        input_file = "{}/{}.json.gz".format(objectnav_episodes, scene_id)
        d = load_dataset(input_file)
        d["episodes"] = episodes
        output_file = "{}/{}.json".format(output_path, scene_id)
        print("Writing at: {}".format(output_file))
        write_json(d, output_file)
        write_gzip(output_file, output_file)


def split_from_file(input_path, split_type, file, output_path):
    dataset = load_dataset(input_path)
    split_json = load_json_dataset(file)

    print("Total episodes: {}".format(len(dataset["episodes"])))
    print("Split type: {}".format(split_type))

    if split_type == "instructions_holdout":
        instructions = split_json["instructions"]
        episodes = []
        for ep in dataset["episodes"]:
            if ep["instruction"]["instruction_text"] in instructions:
                episodes.append(ep)
    elif split_type == "scene_holdout":
        scenes = split_json["scenes"]
        episodes = []
        for ep in dataset["episodes"]:
            if ep["scene_id"] in scenes:
                episodes.append(ep)
    dataset["episodes"] = episodes
    print("Total episodes after filtering: {}".format(len(episodes)))
    write_json(dataset, output_path)
    write_gzip(output_path, output_path)


def split_dataset_and_exclude(input_path, output_path, source_path, num_instruction_episodes=500, num_unseen_init_episodes=2000):
    VISITED_POINT_DICT = {}
    # source_dataset = load_dataset(source_path)
    
    # for episode in source_dataset["episodes"]:
    #     key = str(episode["start_position"])
    #     if key not in VISITED_POINT_DICT.keys():
    #         VISITED_POINT_DICT[key] = 0
    #     VISITED_POINT_DICT[key] += 1

    #     for object_ in episode["objects"]:
    #         point = str(object_["position"])
    #         if point not in VISITED_POINT_DICT.keys():
    #             VISITED_POINT_DICT[point] = 0
    #         VISITED_POINT_DICT[point] += 1

    # print("Total existing episodes: {}".format(len(source_dataset["episodes"])))
    dataset = load_dataset(input_path)
    filtered_episodes = []
    for i, episode in enumerate(dataset["episodes"]):
        episode["episode_id"] = str(i)
        if episode["scene_id"] not in ["TbHJrupSAjP.glb", "JmbYfDe2QKZ.glb"]:
            filtered_episodes.append(episode)

    print("Total episodees: {}, Filtered episodes: {}".format(len(dataset["episodes"]), len(filtered_episodes)))
    dataset["episodes"] = filtered_episodes

    # episodes = []
    # for episode in dataset["episodes"]:
    #     is_valid = True

    #     key = str(episode["start_position"])
    #     if key in VISITED_POINT_DICT.keys():
    #         is_valid = False

    #     for object_ in episode["objects"]:
    #         point = str(object_["position"])
    #         if point in VISITED_POINT_DICT.keys():
    #             is_valid = False
    #     if is_valid:
    #         episodes.append(episode)

    instructions = load_json_dataset("instructions.json")
    inst_seen_episodes = []
    for ep in dataset["episodes"]:
        instruction = ep["instruction"]["instruction_text"].replace(" in", " on")
        if instruction not in instructions:
            inst_seen_episodes.append(ep)

    print("Total filtered episodes: {}".format(len(inst_seen_episodes)))

    unseen_init_episodes = random.sample(inst_seen_episodes, num_unseen_init_episodes)
    unseen_init_episode_ids = [episode["episode_id"] for episode in unseen_init_episodes]

    episodes = []
    for episode in dataset["episodes"]:
        instruction = episode["instruction"]["instruction_text"].replace(" in", " on")
        if episode["episode_id"] not in unseen_init_episode_ids and instruction in instructions:
            episodes.append(episode)
    
    print("Total filtered episodes: {}".format(len(episodes)))

    unseen_instruction_episodes = random.sample(episodes, num_instruction_episodes)
    dataset["episodes"] = unseen_init_episodes
    print("Total episodees unseen inits: {}".format(len(dataset["episodes"])))

    write_json(dataset, output_path)
    write_gzip(output_path, output_path)
    
    dataset["episodes"] = unseen_instruction_episodes
    print("Total episodees unseen instr filter: {}".format(len(dataset["episodes"])))

    write_json(dataset, source_path)
    write_gzip(source_path, source_path)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_data/hits_max_length_1500.json.gz"
    )
    parser.add_argument(
        "--train-data-path", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--eval-data-path", type=str, default="data/datasets/object_rearrangement/v0/eval/eval.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/hit_approvals/dataset/"
    )
    parser.add_argument(
        "--validate", dest='validate', action='store_true'
    )
    parser.add_argument(
        "--validate-before-split", dest='validate_before_split', action='store_true'
    )
    parser.add_argument(
        "--create-split", dest='create_split', action='store_true'
    )
    parser.add_argument(
        "--instruction-holdout", dest='instruction_holdout', action='store_true'
    )
    parser.add_argument(
        "--scene-holdout", dest='scene_holdout', action='store_true'
    )
    parser.add_argument(
        "--objectnav-spath", dest='objectnav_spath', action='store_true'
    )
    parser.add_argument(
        "--split-type", type=str, default='none'
    )
    parser.add_argument(
        "--exclude", dest='exclude', action='store_true'
    )
    
    args = parser.parse_args()

    if args.exclude:
        split_dataset_and_exclude(args.input_path, args.output_path, args.train_data_path)
    elif args.create_split:
        create_split_from_existing_data(args.input_path, args.train_data_path, args.eval_data_path, args.output_path)
    elif args.validate_before_split:
        validate_split_data(args.input_path, args.train_data_path, args.eval_data_path)
    elif args.validate:
        validate_data(args.input_path, args.train_data_path, args.eval_data_path)
    elif args.objectnav_spath:
        convert_objectnav_episodes_to_scene_split(args.input_path, args.output_path)
    elif args.split_type != "none":
        split_from_file(args.input_path, args.split_type, "data/datasets/object_rearrangement/split.json", args.output_path)
    else:
        split_data(args.input_path, args.output_path, args.instruction_holdout, args.scene_holdout)


if __name__ == "__main__":
    main()
