import argparse
import glob

from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset, write_gzip


category_map = {
    "couch": "sofa",
    "toilet": "toilet",
    "bed": "bed",
    "tv": "tv_monitor",
    "potted plant": "plant",
    "chair": "chair"
}

def map_gibson_categories_to_mp3d(path):
    files = glob.glob(path)
    for f in files:
        d = load_dataset(f)
        d["gibson_to_mp3d_category_map"] = category_map

        output_path = f.replace(".gz", "")
        write_json(d, output_path)
        write_gzip(output_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_approvals/dataset/backup/train.json.gz"
    )
    args = parser.parse_args()

    map_gibson_categories_to_mp3d(args.path)

if __name__ == "__main__":
    main()
