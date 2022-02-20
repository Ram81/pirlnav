import json
import os
import requests
import shutil
import subprocess
import urllib

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Process


API_ENDPOINT = "https://fuel.ignitionrobotics.org/1.0"
GET_MODELS_API = "/{}/models?page={}"
DOWNLOAD_API = "/{}/models/{}/{}/files/{}"
FILE_TREE_API = "/{}/models/{}/{}/files"
DOWNLOAD_PATH = "data/google_object_dataset/"
GLB_DIR = "data/google_object_dataset/glb"
NUM_PROCESS = 5


def execute_get_request(url):
    response = requests.get(url)
    return response


def write_response(response, file_path):
    file = open(file_path, "wb")
    for chunk in response.iter_content(chunk_size=8192): 
        file.write(chunk)


def read_json(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def write_json(data, path):
    file = open(path, "w")
    file.write(json.dumps(data))


def get_model_file_tree(model_name, owner, version=1):
    url = API_ENDPOINT + FILE_TREE_API.format(owner, model_name, version)
    return execute_get_request(url)


def download_file(url, output_path):
    response = execute_get_request(url)
    write_response(response, output_path)


def make_directories(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def copy_file(source_path, dest_path):
    shutil.copy2(source_path, dest_path)


def convert_to_glb(path, output_path):
    os.system("obj2gltf -i {} -o {}".format(path, output_path))


def create_object_config(path, output_path):
    asset_name = path.split("/")[-1]
    data = {
        "render_asset": asset_name,
        "use_bounding_box_for_collision": True,
        "requires_lighting": True,
        "margin": 0
    }
    write_json(data, output_path)


def download_object_model(model_name, owner, version=1):
    file_tree_response = get_model_file_tree(model_name, owner, version)
    file_tree_response = file_tree_response.json()

    model_path = DOWNLOAD_PATH + model_name
    make_directories(model_path)
    make_directories(GLB_DIR)

    texture_file_path = ""
    meshes_path = ""
    thumbnail_path = ""
    for files in file_tree_response["file_tree"]:
        name = files["name"]

        if "children" in files.keys():
            children = files["children"]

            if name == "materials":
                children = children[0]["children"]
                for child_object in children:
                    file_path = child_object["path"]
                    output_path = model_path + file_path

                    texture_file_path = output_path

                    directory_path = model_path + "/".join(file_path.split("/")[:-1])
                    make_directories(directory_path)

                    url_encoded_file_path = urllib.parse.quote_plus(file_path[1:])
                    download_url = API_ENDPOINT + DOWNLOAD_API.format(owner, model_name, version, url_encoded_file_path)
                    # print("Object model: {}, Output path: {}".format(model_name, output_path))
                    download_file(download_url, output_path)
            elif name in ["meshes", "thumbnails"]:
                for child_object in children:
                    file_path = child_object["path"]
                    output_path = model_path + file_path

                    directory_path = model_path + "/".join(file_path.split("/")[:-1])
                    make_directories(directory_path)

                    if name == "meshes":
                        meshes_path = directory_path
                    
                    if name == "thumbnails":
                        thumbnail_path = output_path

                    url_encoded_file_path = urllib.parse.quote_plus(file_path[1:])
                    download_url = API_ENDPOINT + DOWNLOAD_API.format(owner, model_name, version, url_encoded_file_path)
                    # print("Object model: {}, Output path: {}".format(model_name, output_path))
                    download_file(download_url, output_path)
        else:
            file_path = files["path"]
            output_path = model_path + file_path

            directory_path = model_path + "/".join(file_path.split("/")[:-1])
            make_directories(directory_path)

            url_encoded_file_path = urllib.parse.quote_plus(file_path[1:])

            download_url = API_ENDPOINT + DOWNLOAD_API.format(owner, model_name, version, url_encoded_file_path)
            # print("Object model: {}, Output path: {}".format(model_name, output_path))
            download_file(download_url, output_path)
    copy_file(texture_file_path, meshes_path)

    object_file = meshes_path + "/model.obj"
    output_glb_path = GLB_DIR + "/{}.glb".format(model_name)
    object_config_path = GLB_DIR + "/{}.object_config.json".format(model_name)
    icon_path = GLB_DIR + "/{}.png".format(model_name)
    convert_to_glb(object_file, output_glb_path)
    create_object_config(output_glb_path, object_config_path)
    copy_file(thumbnail_path, icon_path)


def get_models(owner, page):
    url = API_ENDPOINT + GET_MODELS_API.format(owner, page)
    return execute_get_request(url).json()


def download_all_google_objects(owner="GoogleResearch", start_page=1, max_pages=10):
    for page in tqdm(range(start_page, start_page + max_pages)):
        data = get_models(owner, page)
        print("Page: {}, Objects: {}".format(page, len(data)))
        for idx in range(len(data)):
            model = data[idx]
            model_name = model["name"]
            owner = model["owner"]
            download_object_model(model_name, owner)


if __name__ == "__main__":
    pool = []
    for process_id in range(NUM_PROCESS):
        start_page = process_id * 10 + 1
        p = Process(target=download_all_google_objects, args=("GoogleResearch", start_page, 10, ))
        pool.append(p)
        p.start()
    
    # block until all the threads finish (i.e. block until all function_x calls finish)    
    for t in pool:
        t.join()
        
