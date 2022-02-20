import csv
import gzip
import os
import random
from typing import List

import habitat

MP3D_SCENES_SPLITS_PATH = "habitat/datasets/scenes/scenes_mp3d.csv"
HABITAT_MP3D_SCENES_SPLITS_PATH = (
    "habitat/datasets/scenes/scenes_habitat_mp3d.csv"
)
GIBSON_SCENES_SPLITS_PATH = (
    "habitat/datasets/scenes/scenes_gibson_fullplus.csv"
)

GIBSON_HABITAT_SCENES_SPLITS_PATH = (
    "habitat/datasets/scenes/scenes_gibson_habtiat.csv"
)


def get_mp3d_scenes(
    split: str = "train", scene_template: str = "{scene}"
) -> List[str]:
    scenes = []
    with open(MP3D_SCENES_SPLITS_PATH, newline="") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=",", quotechar="|")
        for row in spamreader:
            if split in row["set"].split() or split == "*":
                scenes.append(scene_template.format(scene=row["id"]))
    return scenes


def get_gibson_scenes(
    split: str = "train", scene_template: str = "{scene}"
) -> List[str]:
    scenes = []
    with open(GIBSON_SCENES_SPLITS_PATH, newline="") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=",", quotechar="|")
        for row in spamreader:
            if row[split] == "1" or split == "*":
                scenes.append(scene_template.format(scene=row["id"]))
    return scenes


def get_habitat_mp3d_scenes(
    split: str = "train", scene_template: str = "{scene}"
) -> List[str]:
    scenes = []
    with open(HABITAT_MP3D_SCENES_SPLITS_PATH, newline="") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter="\t", quotechar="|")
        for row in spamreader:
            if split == "*" or split in row["set"].split():
                scenes.append(
                    scene_template.format(
                        scene=f'{row["folder"]}/{row["glb"]}'
                    )
                )
    return scenes


def get_habitat_gibson_scenes(
    split: str = "train", scene_template: str = "{scene}"
) -> List[str]:
    scenes = []
    with open(GIBSON_HABITAT_SCENES_SPLITS_PATH, newline="") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=",", quotechar="|")
        for row in spamreader:
            if row[split] == "1" or split == "*":
                scenes.append(scene_template.format(scene=row["id"]))
    return scenes


def get_avg_geo_dist(dataset):
    sum_geo_dist = 0
    for episode in dataset.episodes:
        sum_geo_dist += episode.info["geodesic_distance"]

    return sum_geo_dist / len(dataset.episodes)


def generate_sampled_train(
    config_path="datasets/pointnav/gibson.yaml", num_episodes=1000
):
    config = habitat.get_config(config_path, config_dir="habitat-api/configs")
    config.defrost()
    config.DATASET.SPLIT = "train"
    config.freeze()
    print("Dataset is loading.")
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    print(config.DATASET.SPLIT + ": Len episodes: ", len(dataset.episodes))
    dataset.episodes = random.sample(dataset.episodes, num_episodes)
    print("Average geo distance: ", get_avg_geo_dist(dataset))

    json_str = str(dataset.to_json())

    output_dir = "data/datasets/pointnav/gibson/v1/{}_small_2/".format(
        config.DATASET.SPLIT
    )
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    main_dataset_file = "{}/{}_small_2.json.gz".format(
        output_dir, config.DATASET.SPLIT
    )
    with gzip.GzipFile(main_dataset_file, "wb") as f:
        f.write(json_str.encode("utf-8"))

    print("Dataset file: {}".format(main_dataset_file))


def generate_mini_train_splits():
    generate_sampled_train("datasets/pointnav/gibson.yaml")
    # generate_sampled_train("datasets/pointnav/mp3d.yaml")
