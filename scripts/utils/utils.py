import gzip
import json

import matplotlib.pyplot as plt

def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def load_vocab(vocab_path="data/datasets/object_rearrangement/vocab.json"):
    vocab_file = open(vocab_path, "r")
    vocab = json.loads(vocab_file.read())
    return vocab


def get_episode_idx_by_instructions(episodes, instructions):
    episode_idx = []
    for i, episode in enumerate(episodes):
        if episode["instruction"]["instruction_text"] in instructions:
            episode_idx.append(i)
    return episode_idx


def get_episode_idx_by_scene_ids(episodes, scene_ids):
    episode_idx = []
    for i, episode in enumerate(episodes):
        if episode["scene_id"] in scene_ids:
            episode_idx.append(i)
    return episode_idx


def get_episodes_by_episode_ids(episodes, episode_ids):
    filtered_episodes = []
    for episode in episodes:
        if episode["episode_id"] in episode_ids:
            filtered_episodes.append(episode)
    return filtered_episodes


def get_episodes_by_episode_index(episodes, indices):
    filtered_episodes = []
    for i, episode in enumerate(episodes):
        if i in indices:
            filtered_episodes.append(episode)
    return filtered_episodes


def get_unique_scenes(episodes):
    scenes = []
    for ep in episodes:
        scenes.append(ep["scene_id"])
    return list(set(scenes))

def prep_plt(ax=None):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 24
    LARGE_SIZE = 35
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    plt.figure(figsize=(6,4))
    if ax is None:
        ax = plt.gca()
    spine_alpha = 0.3
    ax.spines['right'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    ax.spines['top'].set_alpha(spine_alpha)
    ax.grid(alpha=0.25)
    plt.tight_layout()
