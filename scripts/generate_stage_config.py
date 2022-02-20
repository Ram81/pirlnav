import argparse
import os
import glob
import shutil

from collections import defaultdict
from psiturk_dataset.utils.utils import write_json, load_dataset


def list_files(path):
    file_paths = glob.glob(path)
    return file_paths


def create_stage_config(path):
    files = list_files(path)
    for file_path in files:
        scene_file = file_path.split("/")[-1].split(".")[0]
        print(file_path)
        dir_path = "/".join(file_path.split("/")[:-1])
        stage_config_path = os.path.join(dir_path, scene_file + ".stage_config.json")

        config = {
            "render_asset": "{}.glb".format(scene_file),
            "nav_asset": "{}.navmesh".format(scene_file),
            "semantic_asset": "{}_semantic.ply".format(scene_file),
            "house_filename": "{}.scn".format(scene_file),
            "render mesh": "{}.glb".format(scene_file),
            "nav mesh": "{}.navmesh".format(scene_file),
            "semantic mesh": "{}_semantic.ply".format(scene_file),
            "house filename": "{}.scn".format(scene_file),
            "up":[0,0,1],
            "front":[0,1,0],
            "origin":[0,0,0],
            "scale":[2,2,2],
            "gravity":[0,-9.8,0],
            "margin":0.03,
            "friction_coefficient": 0.3,
            "restitution_coefficient": 0.3,
            "units_to_meters":1.0,
            "requires_lighting": True,
            "rigid object paths": [
                # "/data/objects/",
                # "../test_assets/objects/",
                # "../../test_assets/objects/"
            ]
        }
        write_json(config, stage_config_path)
        print("Stage: {}".format(stage_config_path))


def copy_scenes(src_path, dest_path):
    files = os.listdir(src_path)

    valid_scenes = glob.glob("data/datasets/objectnav_mp3d_v1/train/content/*.json.gz")
    valid_scenes = [s.split("/")[-1].split(".")[0] for s in valid_scenes]
    selected = []
    i = 0 
    for item in files:
        if not item in valid_scenes:
            continue
        scene = item.split("/")[-1]
        item_dest_path = os.path.join(dest_path, scene)
        print(item_dest_path)
        if os.path.exists(item_dest_path):
           shutil.rmtree(item_dest_path)
        src_p = os.path.join(src_path, item)
        shutil.copytree(src_p, item_dest_path)
        selected.append(item)
        i+=1
        # if i == 2:
        #     break
    print("Copied {} scenes".format(len(selected)))


def get_avg_categories():
    valid_scenes = glob.glob("data/datasets/objectnav_mp3d_v1/val_mini/*.json.gz")
    avg_cats = 0
    max_eps = 0
    cnt = 0
    unique_cats = []
    all_eps = 0
    for file_path in valid_scenes:
        d = load_dataset(file_path)
        object_cats = defaultdict(int)
        for ep in d['episodes']:
            object_cats[ep['object_category']] += 1
            max_eps = max(object_cats[ep['object_category']], max_eps)
            if object_cats[ep['object_category']] == 1:
                unique_cats.append(ep['object_category'])
        avg_cats += len(object_cats.keys())
        all_eps += len(d['episodes'])
        cnt+=1
    print("Total scenes: {}".format(len(valid_scenes)))
    print("Average categories per scene: {}, Max ep: {}.\n Unique categories: {}, num episodes: {}".format(avg_cats/cnt, max_eps, len(set(unique_cats)), all_eps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_data"
    )
    parser.add_argument(
        "--dest-path", type=str, default="data/hit_data"
    )
    parser.add_argument(
        "--copy", dest='copy_flag', action='store_true'
    )

    args = parser.parse_args()
    if args.copy_flag:
        copy_scenes(args.path, args.dest_path)
    else:
        create_stage_config(args.path)
    # get_avg_categories()
