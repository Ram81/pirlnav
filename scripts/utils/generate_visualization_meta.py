import argparse
import os
import glob

from scripts.utils.utils import write_json
from tqdm import tqdm


def create_visualization_metadata(path, output_path):
    # files = glob.glob(os.path.join(path, "*mp4"))
    files = glob.glob(path)
    episode_metas = []

    for f in tqdm(files):
        video_id = f.split("/")[-1]
        episode = f.split("/")[-1].split("-")[0]
        scene_id = episode.split("_")[0]
        episode_id = episode.split("_")[1]

        episode_meta = {
            "episodeId": episode,
            "sceneId": scene_id,
            "video": f,
            "task": "all",
            "episodeLength": 1
        }
        episode_metas.append(episode_meta)
    
    write_json(episode_metas, output_path)

    print("Total episodes: {}".format(len(episode_metas)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="replays/demo_1.json.gz"
    )
    args = parser.parse_args()

    create_visualization_metadata(args.path, args.output_path)
