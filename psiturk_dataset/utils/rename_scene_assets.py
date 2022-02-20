import argparse
import glob
import os
import shutil


scene_map = {
    "empty_house": "29hnd4uzFmX",
    "house": "q9vSo1VnCiC",
    "big_house": "i5noydFURQK",
    "big_house_2": "S9hNv5qa7GM",
    "bigger_house": "JeFG25nYj2p",
    "house_4": "zsNo4HB9uLZ",
    "house_5": "TbHJrupSAjP",
    "house_6": "JmbYfDe2QKZ",
    "house_8": "jtcxE69GiFV",
}


def rename_files(path, extension):
    file_paths = glob.glob(path + "/*.{}".format(extension))
    for file_path in file_paths:
        scene = file_path.split("/")[-1].split(".")[0]
        key = scene
        if "semantic" in scene:
            key = "_".join(scene.split("_")[:-1])

        print("scene : {} -- {}".format(scene, key))
        if key not in scene_map.keys():
            continue

        dest_scene = scene_map[key]

        if "semantic" in scene:
            dest_scene = scene.replace(key, dest_scene)

        dir_path = "/".join(file_path.split("/")[:-1])

        dest_path = os.path.join(dir_path, "{}.{}".format(dest_scene, extension))
        print("Copying: {} - {}".format(file_path, dest_path))
        shutil.copy(file_path, dest_path)


def rename_scenes(path):
    extensions = ["glb", "ply", "house", "navmesh", "stage_config.json"]

    for extension in extensions:
        rename_files(path, extension)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/scene_datasets/habitat-test-scenes/"
    )
    args = parser.parse_args()
    rename_scenes(args.path)
