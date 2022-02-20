import argparse
import json
import glob
import shutil

data = []


def copy_file(src, dest):
	shutil.copyfile(src, dest)


def copy_glb_assets(src_path, dest_path):
    glb_files = glob.glob(src_path + "/*.glb")
    for glb_file in glb_files:
        asset_name = glb_file.split("/")[-1]
        asset_name_clipped = asset_name[4:]
        copy_file(glb_file, dest_path + "/" + asset_name_clipped)


def copy_object_configs(src_path, dest_path):
    object_config_files = glob.glob(src_path + "/*.phys_properties.json")
    print(len(object_config_files))
    for object_config_file in object_config_files:
        config_file = object_config_file.split("/")[-1]
        # Read existing physics config
        f = open(object_config_file, "r")
        org_object_config = json.loads(f.read())
        object_config = {}
        object_config["render_asset"] = config_file.split(".")[0] + ".glb"
        object_config["use_bounding_box_for_collision"] = True
        
        if org_object_config.get("scale") is not None:
            object_config["scale"] = org_object_config["scale"]
        if org_object_config.get("margin") is not None:
            object_config["margin"] = org_object_config["margin"]

        config_file = config_file.split(".")
        config_file[1] = "object_config"
        config_file = ".".join(config_file)

        f = open(dest_path + "/" + config_file, "w")
        f.write(json.dumps(object_config))
        print(dest_path + "/" + config_file)


def write_object_meta(src_path):
    glb_files = glob.glob(src_path + "/*.glb")
    for glb_file in glb_files:
        asset_name = glb_file.split("/")[-1][4:]
        asset_file_name = asset_name.split(".")[0]
        data.append({
            "object": " ".join(asset_file_name.split("_")),
            # "objectIcon": "/data/test_assets/objects/{}.png".format(asset_file_name),
            "objectIcon": "/data/test_assets/objects/wood_block.png",
            "objectHandle": "/data/objects/{}.object_config.json".format(asset_file_name),
            "physicsProperties": "test_assets/objects/{}.object_config.json".format(asset_file_name),
            "renderMesh": "test_assets/objects/{}.glb".format(asset_file_name)
        })
    #print(data)
    with open("object-meta.json", "w") as f:
        f.write(json.dumps(data))


def copy_using_object_meta_file(src_path, src_conf_path, dest_path, path="object-meta.json"):
    f = open(path, "r")
    object_meta_list = json.loads(f.read())
    for object_meta in object_meta_list:
        object_glb = object_meta["renderMesh"].split("/")[-1]
        object_config = object_meta["objectHandle"].split("/")[-1]
        dest_name = "_".join(object_meta["object"].split())

        object_config = {}
        object_config["render_asset"] = dest_name + ".glb"
        object_config["use_bounding_box_for_collision"] = True
        object_config["requires_lighting"] = True
        object_config["scale"] = [2.0, 2.0, 2.0]
        object_config["margin"] = 0

        shutil.copy(src_path + "/" + object_glb, dest_path + "/" + dest_name + ".glb")
        with open(dest_path + "/" + dest_name + ".object_config.json", "w") as f:
            f.write(json.dumps(object_config, indent=4))
        data.append({
            "object": object_meta["object"],
            "objectIcon": "/data/test_assets/objects/wood_block.png",
            "objectHandle": "/data/objects/{}.object_config.json".format(dest_name),
            "physicsProperties": "test_assets/objects/{}.object_config.json".format(dest_name),
            "renderMesh": "test_assets/objects/{}.glb".format(dest_name)
        })
    with open("object-meta.json", "w") as f:
        f.write(json.dumps(data))


def get_object_name_map_from_directory(src_path):
    glb_files = glob.glob(src_path + "/*.glb")
    print(len(glb_files))
    objects = []
    for glb_file in glb_files:
        object_name = glb_file.split("/")[-1].split(".")[0]
        object_config = {}
        objects.append({
            "object": object_name,
            "objectIcon": "/data/test_assets/objects/{}.png".format(object_name),
            "objectHandle": "/data/objects/{}.object_config.json".format(object_name),
            "physicsProperties": "test_assets/objects/{}.object_config.json".format(object_name),
            "renderMesh": "test_assets/objects/{}.glb".format(object_name)
        })
    with open("object-meta.json", "w") as f:
        f.write(json.dumps(objects))


def filter_objects_in_directory(src_path, path="object-meta.json"):
    f = open(path)
    data = json.loads(f.read())
    object_name_map = {}
    for object_ in data:
        object_name = object_["objectHandle"].split("/")[-1].split(".")[0]
        object_name_map[object_name] = 1

    glb_files = glob.glob(src_path + "/*")
    print(len(glb_files))
    objects = []
    for glb_file in glb_files:
        file_name = glb_file.split("/")[-1]
        object_name = file_name.split(".")[0]
        if not object_name in object_name_map.keys():    
            print(file_name)
            shutil.move(glb_file, "data/google_objects/" + file_name)


def get_object_name_map(path="object-meta.json"):
    f = open(path, "r")
    data = json.loads(f.read())

    object_name_map = {}
    for obj in data:
        obj_name = obj["objectHandle"].split("/")[-1].split(".")[0]
        object_name_map[obj_name] = obj["object"]
        print("\"{}\": \"{}\"".format(obj_name, obj["object"]))
    print("\n\n")
    print(list(object_name_map.keys()))
    print("\n\n")
    print(list(object_name_map.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-path", type=str, default="../ycb_google_16k/configs_gltf"
    )
    parser.add_argument(
        "--src-object-conf-path", type=str, default="../ycb_google_16k/configs"
    )
    parser.add_argument(
        "--dest-path", type=str, default="../test"
    )
    
    args = parser.parse_args()
    # copy_glb_assets(args.src_glb_path, args.dest_path)
    # copy_object_configs(args.src_object_conf_path, args.dest_path)
    # write_object_meta(args.src_glb_path)
    # copy_using_object_meta_file(args.src_glb_path, args.src_object_conf_path, args.dest_path)
    # get_object_name_map()
    # get_object_name_map_from_directory("../habitat-sim/data/objects")
    copy_object_configs(args.src_path, args.dest_path)

