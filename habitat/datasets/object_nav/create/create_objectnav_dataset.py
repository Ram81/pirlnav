import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import numpy as np
import pydash
import tqdm

import habitat
import habitat_sim
from habitat.datasets.object_nav.create.object_nav_generator import (
    build_goal,
    generate_objectnav_episode,
)
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector

num_episodes_per_scene = int(5e4)

whitelist_classes = [
    {"children": [], "id": 1, "name": "backpack", "parents": []},
    {"children": [], "id": 3, "name": "basket", "parents": []},
    {"children": [], "id": 4, "name": "bathtub", "parents": []},
    {"children": [], "id": 6, "name": "beanbag", "parents": []},
    {"children": [], "id": 7, "name": "bed", "parents": []},
    {"children": [], "id": 8, "name": "bench", "parents": []},
    {"children": [], "id": 9, "name": "bike", "parents": []},
    {"children": [], "id": 10, "name": "bin", "parents": []},
    {"children": [], "id": 11, "name": "blanket", "parents": []},
    {"children": [], "id": 13, "name": "book", "parents": []},
    {"children": [], "id": 14, "name": "bottle", "parents": []},
    {"children": [], "id": 15, "name": "box", "parents": []},
    {"children": [], "id": 16, "name": "bowl", "parents": []},
    {"children": [], "id": 17, "name": "camera", "parents": []},
    {"children": [], "id": 18, "name": "cabinet", "parents": []},
    {"children": [], "id": 19, "name": "candle", "parents": []},
    {"children": [], "id": 20, "name": "chair", "parents": []},
    {"children": [], "id": 21, "name": "chopping-board", "parents": []},
    {"children": [], "id": 22, "name": "clock", "parents": []},
    {"children": [], "id": 23, "name": "cloth", "parents": []},
    {"children": [], "id": 24, "name": "clothing", "parents": []},
    {"children": [], "id": 25, "name": "coaster", "parents": []},
    {"children": [], "id": 26, "name": "comforter", "parents": []},
    {"children": [], "id": 27, "name": "computer-keyboard", "parents": []},
    {"children": [], "id": 28, "name": "cup", "parents": []},
    {"children": [], "id": 29, "name": "cushion", "parents": []},
    {"children": [], "id": 30, "name": "curtain", "parents": []},
    {"children": [], "id": 31, "name": "ceiling", "parents": []},
    {"children": [], "id": 32, "name": "cooktop", "parents": []},
    {"children": [], "id": 33, "name": "countertop", "parents": []},
    {"children": [], "id": 34, "name": "desk", "parents": []},
    {"children": [], "id": 35, "name": "desk-organizer", "parents": []},
    {"children": [], "id": 36, "name": "desktop-computer", "parents": []},
    {"children": [], "id": 37, "name": "door", "parents": []},
    {"children": [], "id": 38, "name": "exercise-ball", "parents": []},
    {"children": [], "id": 39, "name": "faucet", "parents": []},
    {"children": [], "id": 40, "name": "floor", "parents": []},
    {"children": [], "id": 41, "name": "handbag", "parents": []},
    {"children": [], "id": 42, "name": "hair-dryer", "parents": []},
    {"children": [], "id": 44, "name": "indoor-plant", "parents": []},
    {"children": [], "id": 45, "name": "knife-block", "parents": []},
    {"children": [], "id": 47, "name": "lamp", "parents": []},
    {"children": [], "id": 48, "name": "laptop", "parents": []},
    {"children": [], "id": 50, "name": "mat", "parents": []},
    {"children": [], "id": 51, "name": "microwave", "parents": []},
    {"children": [], "id": 52, "name": "monitor", "parents": []},
    {"children": [], "id": 53, "name": "mouse", "parents": []},
    {"children": [], "id": 54, "name": "nightstand", "parents": []},
    {"children": [], "id": 55, "name": "pan", "parents": []},
    {"children": [], "id": 57, "name": "paper-towel", "parents": []},
    {"children": [], "id": 58, "name": "phone", "parents": []},
    {"children": [], "id": 59, "name": "picture", "parents": []},
    {"children": [], "id": 60, "name": "pillar", "parents": []},
    {"children": [], "id": 61, "name": "pillow", "parents": []},
    {"children": [], "id": 62, "name": "pipe", "parents": []},
    {"children": [], "id": 64, "name": "plate", "parents": []},
    {"children": [], "id": 65, "name": "pot", "parents": []},
    {"children": [], "id": 66, "name": "rack", "parents": []},
    {"children": [], "id": 67, "name": "refrigerator", "parents": []},
    {"children": [], "id": 68, "name": "remote-control", "parents": []},
    {"children": [], "id": 69, "name": "scarf", "parents": []},
    {"children": [], "id": 70, "name": "sculpture", "parents": []},
    {"children": [], "id": 73, "name": "shower-stall", "parents": []},
    {"children": [], "id": 74, "name": "sink", "parents": []},
    {"children": [], "id": 76, "name": "sofa", "parents": []},
    {"children": [], "id": 77, "name": "stair", "parents": []},
    {"children": [], "id": 78, "name": "stool", "parents": []},
    {"children": [], "id": 79, "name": "switch", "parents": []},
    {"children": [], "id": 80, "name": "table", "parents": []},
    {"children": [], "id": 83, "name": "tissue-paper", "parents": []},
    {"children": [], "id": 84, "name": "toilet", "parents": []},
    {"children": [], "id": 85, "name": "toothbrush", "parents": []},
    {"children": [], "id": 86, "name": "towel", "parents": []},
    {"children": [], "id": 87, "name": "tv-screen", "parents": []},
    {"children": [], "id": 89, "name": "umbrella", "parents": []},
    {"children": [], "id": 90, "name": "utensil-holder", "parents": []},
    {"children": [], "id": 91, "name": "vase", "parents": []},
    {"children": [], "id": 96, "name": "wardrobe", "parents": []},
    {"children": [], "id": 98, "name": "rug", "parents": []},
    {"children": [], "id": 100, "name": "bag", "parents": []},
]

name_to_id = (
    pydash.chain()
    .group_by("name")
    .map_values(lambda v: v[0])
    .map_values("id")(whitelist_classes)
)
id_to_name = {v: k for k, v in name_to_id.items()}


def _generate_fn(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = ["SEMANTIC_SENSOR"]
    cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    cfg.freeze()

    with open(osp.join(osp.dirname(scene), "info_semantic.json"), "r") as f:
        semantics = json.load(f)

    objects = semantics["objects"]
    object_by_class = pydash.group_by(objects, "class_id")
    object_by_class = {
        k: v
        for k, v in object_by_class.items()
        if k in pydash.map_(whitelist_classes, "id")
    }

    if False:
        for to_find in [
            "desk",
            "refrigerator",
            "toilet",
            "shower-stall",
            "bed",
            "sculpture",
        ]:
            if name_to_id[to_find] in object_by_class.keys():
                print("found {}".format(to_find))

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    goals_by_class = {}
    for k, v in tqdm.tqdm(object_by_class.items()):
        goals = []
        for o in tqdm.tqdm(v, leave=False):
            goals.append(
                build_goal(
                    sim,
                    o["id"],
                    o["class_id"],
                    quat_rotate_vector(
                        quat_from_two_vectors(
                            np.array([0, 0, -1]), habitat_sim.geo.GRAVITY
                        ),
                        np.array(o["oriented_bbox"]["abb"]["center"]),
                    ),
                )
            )
        goals = [g for g in goals if g is not None]

        if len(goals) == 0:
            continue

        goals_by_class[k] = goals

    if len(goals_by_class) == 0:
        return None

    dset = habitat.datasets.make_dataset("ObjectNav-v1")
    eps_per_obj = 500
    with tqdm.tqdm(total=len(goals_by_class) * eps_per_obj) as pbar:
        for goals in goals_by_class.values():
            try:
                for ep in generate_objectnav_episode(
                    sim, goals, num_episodes=eps_per_obj
                ):
                    dset.episodes.append(ep)
                    pbar.update()
            except RuntimeError:
                pbar.update()

    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
    fname = f"./data/datasets/objectnav/replica/v2/train/content/{scene_key}.json.gz"
    os.makedirs(osp.dirname(fname), exist_ok=True)
    with gzip.open(fname, "wt") as f:
        f.write(dset.to_json())

    sim.close()


if __name__ == "__main__":
    mp_ctx = multiprocessing.get_context("forkserver")
    scenes = glob.glob(
        "./data/scene_datasets/replica_dataset/*/habitat/mesh_semantic.ply"
    )
    blacklist = {"room_1", "room_2", "hotel_0", "office_0", "office_1"}
    blacklist = blacklist.union({"apartment_0", "hotel_0", "aparment_2"})
    blacklist = {}
    scenes = [s for s in scenes if not any(k in s for k in blacklist)]
    with mp_ctx.Pool(20) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()
