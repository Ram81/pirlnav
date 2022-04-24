import argparse
import os
import glob
import zipfile

from scripts.utils.utils import load_dataset

DATASET_PREFIX = "/srv/datasets/habitat-sim-datasets/"


def list_files(path):
    return glob.glob(path)


def get_scene_dir(path):
    dataset = load_dataset(path)
    scene_path = "/".join(dataset["episodes"][0]["scene_id"].split("/")[:-1])
    return "{}{}".format(DATASET_PREFIX, scene_path)


def zip_scenes(dataset_path, zip_output_path):
    files = list_files(dataset_path)

    scene_dataset_zip = zipfile.ZipFile(zip_output_path, "w")

    print("Total scenes: {}".format(len(files)))
    for scene_data_path  in files:
        print("HM3D scene episode: {}".format(scene_data_path))
        scene_dir = get_scene_dir(scene_data_path)
        print("Scene directory: {}".format(scene_data_path))

        scene_dataset_zip.write(scene_dir)
        scene_files = list_files(os.path.join(scene_dir, "*"))

        for scene_file in scene_files:
            if scene_file.endswith(".jpg"):
                continue
            print("Scene file: {}".format(scene_file))
            scene_dataset_zip.write(scene_file)
    scene_dataset_zip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, default="data/datasets/objectnav_hm3d_v1/train/content/*json.gz"
    )
    parser.add_argument(
        "--zip-output-path", type=str, default="hm3d_train.zip"
    )
    args = parser.parse_args()

    zip_scenes(args.dataset_path, args.zip_output_path)
