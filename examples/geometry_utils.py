import argparse
import habitat
import os
import torch
import quaternion
import numpy as np
import torch.nn.functional as F

from PIL import Image
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from scripts.utils.utils import write_json, load_json_dataset

from mmdet.apis import init_detector, inference_detector # needs MMDetection library
# from open3d.geometry import create_point_cloud_from_depth_image
from habitat.utils.geometry_utils import (
    quaternion_from_coeff
)


config = habitat.get_config("configs/tasks/objectnav_mp3d_il.yaml")


def make_videos(observations_list, output_prefix, ep_id, fps=5):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix, fps=fps)


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def get_detector(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device)
    return model


def get_device():
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return device


def create_pointcloud_from_depth(depth, intrinsic_matrix, extrinsic_matrix):
    point_cloud = create_point_cloud_from_depth_image(depth, intrinsic_matrix)
    return point_cloud

def transform_depth_image_to_reference_frame(config, depth_1, cameras):
    W = config.SIMULATOR.DEPTH_SENSOR.WIDTH
    H = config.SIMULATOR.DEPTH_SENSOR.HEIGHT

    hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV

    K = np.array([
    [1 / np.tan(hfov / 2.), 0., 0., 0.],
    [0., 1 / np.tan(hfov / 2.), 0., 0.],
    [0., 0.,  1, 0],
    [0., 0., 0, 1]])

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,W))
    depth = depth_1.reshape(1,W,W)
    xs = xs.reshape(1,W,W)
    ys = ys.reshape(1,W,W)

    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    # Now load in the cameras, are in the format camera --> world
    # Camera 1:
    quaternion_0 = cameras[0].sensor_states['depth'].rotation
    translation_0 = cameras[0].sensor_states['depth'].position
    # quaternion_0 = cameras[0]["sensor_states"]['depth']["rotation"]
    # translation_0 = cameras[0]["sensor_states"]['depth']["position"]
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0
    T_world_camera0[0:3,3] = translation_0
    
    print("cam 0: {} {}".format(quaternion_0, translation_0))

    # Camera 2:
    translation_1 = cameras[1].sensor_states['depth'].position
    quaternion_1 = cameras[1].sensor_states['depth'].rotation
    # translation_1 = cameras[1]["sensor_states"]['depth']["position"]
    # quaternion_1 = cameras[1]["sensor_states"]['depth']["rotation"]
    rotation_1 = quaternion.as_rotation_matrix(quaternion_1)
    T_world_camera1 = np.eye(4)
    T_world_camera1[0:3,0:3] =  rotation_1
    T_world_camera1[0:3,3] = translation_1
    print("cam 1: {} {}".format(quaternion_1, translation_1))

    # Invert to get world --> camera
    T_camera1_world = np.linalg.inv(T_world_camera1)

    # Transformation matrix between views
    # Aka the position of camera0 in camera1's coordinate frame
    T_camera1_camera0 = np.matmul(T_camera1_world, T_world_camera0)

    # Finally transform actual points
    xy_c1 = np.matmul(T_camera1_camera0, xy_c0)
    xy_newimg = np.matmul(K, xy_c1)

    # Normalize by negative depth
    xys_newimg = xy_newimg[0:2,:] / -xy_newimg[2:3,:]
    print("xys", xys_newimg.shape)
    # Flip back to y-down to match array indexing
    xys_newimg[1] *= -1
    return torch.tensor(xy_newimg).view(4, W, W).numpy()

def get_extrinsic_matrix(camera):
    translation_1 = camera.sensor_states['depth'].position
    quaternion_1 = camera.sensor_states['depth'].rotation
    # translation_1 = cameras[1]["sensor_states"]['depth']["position"]
    # quaternion_1 = cameras[1]["sensor_states"]['depth']["rotation"]
    rotation_1 = quaternion.as_rotation_matrix(quaternion_1)
    T_world_camera1 = np.eye(4)
    T_world_camera1[0:3,0:3] =  rotation_1
    T_world_camera1[0:3,3] = translation_1
    return T_world_camera1
