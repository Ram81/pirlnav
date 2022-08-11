import argparse
import habitat
import attr
import numpy as np
import glob
import os

from PIL import Image
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from scripts.parsing.parse_objectnav_dataset import write_gzip, write_json
from scripts.utils.utils import load_dataset

config = habitat.get_config("configs/tasks/objectnav_mp3d_il.yaml")


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def euclidean_distance(position_a, position_b):
    return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)


def compute_goal_distance_to_view_points(sim, object_goals):
    view_points_within_1m = 0
    for goal in object_goals:
        goal_position = goal.position
        for view_point in goal.view_points:
            view_point_position = view_point.agent_state.position

            # dist = sim.geodesic_distance(goal_position, view_point_position)
            dist = euclidean_distance(goal_position, view_point_position)
            if dist == np.inf:
                continue
            
            view_point.within_1m = bool(dist < 1.5)
            view_points_within_1m += (dist < 1.0)
    
    check_vp = 0
    for goal in object_goals:
        for view_point in goal.view_points:
            check_vp += view_point.within_1m
    
    print("Points within 1m: {} - {}".format(view_points_within_1m, check_vp))
    object_goals_json = [attr.asdict(object_goal) for object_goal in object_goals]
    return object_goals_json


def augment_viewpoints(
    cfg,
    num_episodes=None,
    label="original"
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS  
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        goals_by_category = {}
        for ep_id in range(num_episodes):
            obs = env.reset()
            obs = env.step(action=1)

            info = env.get_metrics()
            frame = observations_to_image(obs, info)
            save_image(frame, "{}_{}_{}.png".format(label, ep_id, 0))

            episode = env.current_episode
            object_goals = compute_goal_distance_to_view_points(env.sim, episode.goals)
            print("end {}".format(episode.scene_id))

            scene_id = episode.scene_id.split("/")[-1]
            category_id = "{}_{}".format(scene_id, episode.object_category)
            goals_by_category[category_id] = object_goals

        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success/num_episodes, total_success, num_episodes))
        return goals_by_category


def augment_viewpoints_for_all_scenes(input_path, output_path, cfg, num_episodes, label):
    scenes = glob.glob(os.path.join(input_path, "*json.gz"))

    scene_ids = [f.split("/")[-1].split(".")[0] for f in scenes]

    for scene, scene_file in zip(scene_ids, scenes):
        # if "b3WpMbPFB6q" not in scene_file:
        #     continue
        cfg.defrost()
        cfg.DATASET.DATA_PATH = scene_file
        cfg.DATASET.TYPE = "ObjectNav-v1"
        cfg.TASK.SENSORS = ["OBJECTGOAL_SENSOR"]
        cfg.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        print(scene_file, scene)
        cfg.freeze()

        d = load_dataset(scene_file)
        print(len(d['episodes']))

        goals_by_category = augment_viewpoints(cfg, num_episodes, label)
        print(goals_by_category.keys(), d["goals_by_category"].keys())
        d["goals_by_category"] = goals_by_category

        if output_path is not None:
            scene_output_path = os.path.join(output_path, "{}.json".format(scene))
            print("Augmented view points for scene: {}".format(scene_output_path))
            write_json(d, scene_output_path)
            write_gzip(scene_output_path, scene_output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default=None
    )
    parser.add_argument(
        "--label", type=str, default="original"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10000
    )
    parser.add_argument(
        "--show-filtered", action="store_true", dest="show_filtered"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS_WITHIN_1M = args.show_filtered
    cfg.freeze()

    augment_viewpoints_for_all_scenes(args.path, args.output_path, cfg, args.num_episodes, args.label)


if __name__ == "__main__":
    main()
