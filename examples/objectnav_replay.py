import open3d as o3d
import argparse
import habitat
import os
import torch
import quaternion
import numpy as np
import torch.nn.functional as F
import cv2
import random

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


def create_pointcloud_from_depth(rgb, depth, intrinsic_matrix, extrinsic_matrix):
    MIN_DEPTH = 0.5
    MAX_DEPTH = 5.0
    #print("depth:{}".format(depth.shape))
    height, width = depth.shape[0], depth.shape[1]
    # depth = ((depth * (MAX_DEPTH - MIN_DEPTH)) + MIN_DEPTH) * 100.0
    depth *= 100.0
    depth = np.where(depth == 0, 0.0001, depth)
    #print(depth.min(), depth.max())
    depth = o3d.geometry.Image(depth)
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, depth_scale=1.0, depth_trunc=1000000.0) #, project_valid_depth_only=False)
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # print("pcd shape: {}".format(np.asarray(point_cloud.points).shape))
    return point_cloud, np.asarray(point_cloud.points).reshape(height, width, 3)


def get_extrinsic_matrix(camera):
    translation_1 = camera["sensor_states"]['depth']["position"]
    quaternion_1 = camera["sensor_states"]['depth']["rotation"]
    rotation_1 = quaternion.as_rotation_matrix(quaternion_1)
    T_world_camera1 = np.eye(4)
    T_world_camera1[0:3,0:3] =  rotation_1
    T_world_camera1[0:3,3] = translation_1
    return T_world_camera1


def get_intrinsic_matrix(cfg):
    height = cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT
    width = cfg.SIMULATOR.DEPTH_SENSOR.WIDTH
    hfov = cfg.SIMULATOR.DEPTH_SENSOR.HFOV

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    fx = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    fy = (height / 2.) / np.tan(np.deg2rad(hfov / 2.))

    intrinsics.set_intrinsics(width, height, fx, fy, xc, zc)
    return intrinsics


def run_detector_inference(detector, observations, point_cloud, filtered_objects):
    result = inference_detector(detector, observations["rgb"])

    nodes = np.concatenate(result[2], axis=0)
    bbox_shape = np.concatenate(result[0])

    non_zero_classes = []
    for i in range(len(result[0])):
        if result[0][i].shape[0] > 0:
            non_zero_classes.append(i)
    class_idxs = np.array(non_zero_classes)
    for i in range(len(result[0])):
        if i not in filtered_objects:
            result[0][i] = np.zeros((0, 5))

    graph = create_graph_from_detector_results(result, point_cloud)
    deduped_graph = deduplicate_local_graph(graph, detector.CLASSES)
    return nodes, deduped_graph, result


def get_point_cloud_overlap(pc_1, pc_2):
    point_cloud_g = o3d.geometry.PointCloud()
    point_cloud_g.points = o3d.utility.Vector3dVector(pc_1.reshape(-1, 3))

    point_cloud_l = o3d.geometry.PointCloud()
    point_cloud_l.points = o3d.utility.Vector3dVector(pc_2.reshape(-1, 3))

    overlap = point_cloud_g.compute_point_cloud_distance(point_cloud_l)
    overlap = np.asarray(overlap)

    return overlap.mean()


def update_global_graph(global_graph, local_graph, objects, merge_dist_thr=10.0):
    for i in range(len(global_graph)):
        global_graph_nodes = global_graph[i]
        local_graph_nodes = local_graph[i]
        # print("objects: {}, walls g: {}, l: {}".format(objects[i], len(global_graph_nodes), len(local_graph_nodes)))

        # Ignore undetected classes. TODO: Handle new classes from local graph. This is just for testing
        if len(local_graph_nodes) == 0:
            # No new nodes in current frame
            continue
        elif len(global_graph_nodes) == 0:
            # No new nodes in global graph - so add all new nodes from current frame
            global_graph[i] = local_graph_nodes
        else:
            is_new_node = [1 for _ in range(len(local_graph_nodes))]
            for j in range(len(global_graph_nodes)):
                for k in range(len(local_graph_nodes)):
                    bbox_g, points_g, _ = global_graph_nodes[j]
                    bbox_l, points_l, _ = local_graph_nodes[k]

                    point_cloud_g = o3d.geometry.PointCloud()
                    point_cloud_g.points = o3d.utility.Vector3dVector(points_g.reshape(-1, 3))

                    point_cloud_l = o3d.geometry.PointCloud()
                    point_cloud_l.points = o3d.utility.Vector3dVector(points_l.reshape(-1, 3))

                    overlap = point_cloud_g.compute_point_cloud_distance(point_cloud_l)
                    overlap = np.asarray(overlap)

                    if overlap.mean() <= merge_dist_thr:
                        is_new_node[k] = 0
            new_nodes = []
            for ii, node in enumerate(local_graph_nodes):
                if is_new_node[ii]:
                    new_nodes.append(node)
            if len(new_nodes) > 0:
                global_graph[i].extend(new_nodes)
    return global_graph


def deduplicate_local_graph(local_graph, objects, merge_dist_thr=10.0):
    updated_graph = []
    for i in range(len(local_graph)):
        updated_graph.append([])
        local_graph_nodes = local_graph[i]
        # print("objects: {}, walls g: {}, l: {}".format(objects[i], len(global_graph_nodes), len(local_graph_nodes)))

        # Ignore undetected classes. TODO: Handle new classes from local graph. This is just for testing
        if len(local_graph_nodes) == 0:
            # No new nodes in current frame
            continue
        else:
            is_new_node = [1 for _ in range(len(local_graph_nodes))]
            for j in range(len(local_graph_nodes)):
                # Ignore already identified duplicate nodes
                if not is_new_node[j]:
                    continue
                for k in range(len(local_graph_nodes)):
                    if j == k:
                        continue
                    bbox_g, points_g, _ = local_graph_nodes[j]
                    bbox_l, points_l, _ = local_graph_nodes[k]

                    point_cloud_g = o3d.geometry.PointCloud()
                    point_cloud_g.points = o3d.utility.Vector3dVector(points_g.reshape(-1, 3))

                    point_cloud_l = o3d.geometry.PointCloud()
                    point_cloud_l.points = o3d.utility.Vector3dVector(points_l.reshape(-1, 3))

                    overlap = point_cloud_g.compute_point_cloud_distance(point_cloud_l)
                    overlap = np.asarray(overlap)

                    if overlap.mean() <= merge_dist_thr:
                        is_new_node[k] = 0
            new_nodes = []
            for ii, node in enumerate(local_graph_nodes):
                if is_new_node[ii]:
                    new_nodes.append(node)
            if len(new_nodes) > 0:
                updated_graph[i].extend(new_nodes)
    return updated_graph


def log_each_cat_nodes(global_graph, classes):
    total_nodes = 0
    for i in range(len(global_graph)):
        total_nodes += len(global_graph[i])
        if len(global_graph[i]) > 0:
            print("class: {}, nodes: {}".format(classes[i], len(global_graph[i])))
    print("Total nodes: {}".format(total_nodes))
    return total_nodes


def create_graph_from_detector_results(result, point_cloud):
    graph = []
    for i in range(len(result[0])):
        graph.append([])
        if result[0][i].shape[0] != 0:
            # print("bbiox: {}".format(result[0][i]))
            for node in result[0][i]:
                if node[4] < 0.5:
                    continue
                node = node.astype(np.int32)
                # score thresholding
                x_0, y_0 = node[0], node[1]
                x_n, y_n = node[2], node[3]
                node_point_cloud = point_cloud[y_0:y_n, x_0:x_n]
                graph[i].append([node, node_point_cloud, result[2][i]])
    return graph


def draw_graph(graph, img_size, objects):
    img = np.zeros(img_size, dtype=np.uint8)
    img.fill(255)

    all_x_ys = []
    pcs = []
    node_count = 0
    for i in range(len(graph)):
        nodes = graph[i]
        object_name = objects[i]
        for node in nodes:
            bbox, pc, _ = node
            x_1, y_1 = bbox[0], bbox[1]
            cv2.putText(img, object_name, (x_1 - 1, y_1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34,200,200), 3)
            cv2.circle(img, (x_1, y_1), 10, (0,0,255), -1)
            all_x_ys.append((x_1, y_1))
            pcs.append(pc)
            node_count += 1

    #all_x_ys = random.sample(all_x_ys, min(len(all_x_ys), 10))
    edges = 0
    for i in range(len(all_x_ys)):
        for j in range(len(all_x_ys)):
            if i != j:
                dist = get_point_cloud_overlap(pcs[i], pcs[j])

                if dist < 20:
                    cv2.line(img, all_x_ys[i], all_x_ys[j], (34,200,93), 1)
                    edges += 1

    cv2.putText(img, "Nodes: {}, Edges: {}".format(node_count, edges), (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34,200,200), 3)
    return img


def run_reference_replay(
    cfg,
    num_episodes=None,
    output_prefix=None,
    append_instruction=False,
    save_videos=False,
    save_step_image=False,
    detector_config="",
    detector_checkpoint=None,
):
    # Set up semantic predictor/detector
    device = get_device()
    detector = get_detector(detector_config, detector_checkpoint, device)
    filtered_objects = load_json_dataset("configs/detector/filtered_objects_mmdet.json")
    print("wall: {}".format(1389 in filtered_objects))

    write_json(detector.CLASSES, "configs/detector/all_objects.json")

    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    intrinsic_matrix = get_intrinsic_matrix(cfg)
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        episode_meta = []
        avg_graph_size = 0
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            point_clouds = []
            obs = env.reset()

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode

            if save_step_image:
                os.makedirs("demos/trajectory_{}".format(ep_id), exist_ok=True)
            
            rgbs =[]
            depths = []
            cameras = []

            rgbs.append(obs["rgb"])
            depths.append(obs["depth"])
            agent_state = env._sim.get_agent_state(0)
            rotation_world_agent = agent_state.rotation
            rotation_world_start = quaternion_from_coeff(episode.start_rotation)

            print("actual agent state: {}, agent pose :{}".format(agent_state.sensor_states, agent_state.position))
            agent_state = {
                "sensor_states": {
                    "depth": {
                        "rotation": rotation_world_start.inverse() * rotation_world_agent,
                        "position": obs["gps"] + [0.0, 1.31, 0.0]
                    }
                }
            }
            cameras.append(agent_state)
            extrinsic_matrix = get_extrinsic_matrix(agent_state)
            pcd, mapped_pcd = create_pointcloud_from_depth(obs["rgb"], obs["depth"], intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)
            point_clouds.append(mapped_pcd)

            nodes, graph, result = run_detector_inference(detector, obs, mapped_pcd, filtered_objects)
            global_graph = graph
            det_frame = detector.show_result(obs["rgb"], (result[0], None), score_thr=0.5)
            graph_frame = draw_graph(global_graph, det_frame.shape, detector.CLASSES)
            
            # Build global graph
            print("-------before update-----------")
            total_nodes = log_each_cat_nodes(global_graph, detector.CLASSES)

            frame = observations_to_image({"rgb": det_frame, "graph": graph_frame}, {})
            if save_step_image:
                save_image(frame, "trajectory_{}/demo_{}_{}.png".format(ep_id, ep_id, 0))

            for step_id, data in enumerate(env.current_episode.reference_replay[step_index:]):
                action = possible_actions.index(data.action)
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)
                agent_state = env._sim.get_agent_state(0)

                rgbs.append(observations["rgb"])
                depths.append(observations["depth"])

                agent_state = env._sim.get_agent_state(0)
                rotation_world_agent = agent_state.rotation
                rotation_world_start = quaternion_from_coeff(episode.start_rotation)

                agent_state = {
                    "sensor_states": {
                        "depth": {
                        "rotation": rotation_world_start.inverse() * rotation_world_agent,
                        "position": observations["gps"] + [0.0, 1.31, 0.0]
                        }
                    }
                }
                cameras.append(agent_state)
                extrinsic_matrix = get_extrinsic_matrix(agent_state)
                pcd, mapped_pcd = create_pointcloud_from_depth(observations["rgb"], observations["depth"], intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)
                point_clouds.append(mapped_pcd)

                nodes, graph, result = run_detector_inference(detector, observations, mapped_pcd, filtered_objects)
                local_graph = graph

                det_frame = detector.show_result(observations["rgb"], (result[0], None), score_thr=0.5)
 
                global_graph = update_global_graph(global_graph, local_graph, detector.CLASSES)
                graph_frame = draw_graph(global_graph, det_frame.shape, detector.CLASSES)

                print("Step id: {}".format(step_id))

                info = env.get_metrics()
                frame = observations_to_image({"rgb": det_frame, "graph": graph_frame}, info)

                if append_instruction:
                    frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                if save_step_image:
                    save_image(frame, "trajectory_{}/demo_{}_{}.png".format(ep_id, ep_id, step_id + 1))

                observation_list.append(frame)

                if action_name == "STOP":
                    break
                if step_id == 100:
                    break
            
            print("\n\n-------after update-----------")
            total_nodes = log_each_cat_nodes(global_graph, detector.CLASSES)

            merged_point_cloud = []
            for i in range(len(global_graph)):
                if len(global_graph[i]) == 0:
                    continue
                for bbox_, node, visual_feats in global_graph[i]:
                    merged_point_cloud.append(node.reshape(-1, 3))
            merged_point_cloud = np.concatenate(merged_point_cloud, axis=0)
            np.save("demos/point_cloud_{}.npy".format(ep_id), merged_point_cloud)
            print(merged_point_cloud.shape)

            avg_graph_size += total_nodes

            if save_videos:
                make_videos([observation_list], output_prefix, ep_id)
            print("Total reward: {}, Success: {}, Steps: {}, Attempts: {}".format(total_reward, info["success"], len(episode.reference_replay), episode.attempts))
            print(info)

            if len(episode.reference_replay) <= 500 and episode.attempts == 1:
                total_success += info["success"]
                spl += info["spl"]

            episode_meta.append({
                "scene_id": episode.scene_id,
                "episode_id": episode.episode_id,
                "metrics": info,
                "steps": len(episode.reference_replay),
                "attempts": episode.attempts,
                "object_category": episode.object_category
            })

        print("Avg. graph size: {}".format(avg_graph_size / num_episodes))
        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success/num_episodes, total_success, num_episodes))

        output_path = os.path.join(os.path.dirname(cfg.DATASET.DATA_PATH), "replay_meta.json")
        write_json(episode_meta, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="demo"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10000
    )
    parser.add_argument(
        "--append-instruction", dest="append_instruction", action="store_true"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000
    )
    parser.add_argument(
        "--save-videos", dest="save_videos", action="store_true"
    )
    parser.add_argument(
        "--save-step-image", dest="save_step_image", action="store_true"
    )
    parser.add_argument(
        "--detector-config", type=str, default="configs/detector/mask_rcnn_r50_270cat.py"
    )
    parser.add_argument(
        "--detector-checkpoint", type=str, default=None
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.DATASET.MAX_EPISODE_STEPS = args.max_steps
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    cfg.TASK.GPS_SENSOR.DIMENSIONALITY = 3
    cfg.freeze()

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix,
        append_instruction=args.append_instruction,
        save_videos=args.save_videos,
        save_step_image=args.save_step_image,
        detector_config=args.detector_config,
        detector_checkpoint=args.detector_checkpoint,
    )


if __name__ == "__main__":
    main()
