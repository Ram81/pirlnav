import argparse
import os
import glob

from collections import defaultdict
from psiturk_dataset.utils.utils import load_dataset, write_json, write_gzip

def convert_s_path_objectnav_episodes_to_dataset(path, output_path):
    scene_ep_map = defaultdict(list)
    files = glob.glob(path + "/*json.gz")
    for file in files:
        print("loading: {}".format(file))
        d = load_dataset(file)
        print("loaded: {}".format(len(d['episodes'])))
        scene_id = ""
        for ep in d['episodes']:
            scene_id = ep['scene_id'].split("/")[-1]
            ep['scene_id'] = "mp3d/" + scene_id.split(".")[0] + "/" + scene_id
            ep["goals"] = []
            scene_ep_map[scene_id].extend(ep)
        print("Scene: {}".format(scene_id))
        break

    files = glob.glob("data/datasets/objectnav_mp3d_v1/train/content/*json.gz")
    for file in files:
        d = load_dataset(file)
        scene_id = file.split("/")[-1].split(".")[0] + ".glb"
        d['episodes'] = scene_ep_map[scene_id]
        o_path = os.path.join(output_path, "{}.json".format(scene_id))
        print("Num episodes: {}".format(len(d['episodes'])))
        if len(d['episodes']) > 0:
            write_json(d, o_path)
            write_json(o_path, o_path)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_approvals/hits_max_length_1500.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sample_hits.json"
    )
    args = parser.parse_args()
    convert_s_path_objectnav_episodes_to_dataset(args.input_path, args.output_path)
