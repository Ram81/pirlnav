import argparse
import habitat
import os
import torch
import numpy as np

from PIL import Image
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from scripts.parsing.parse_objectnav_dataset import write_json
from scripts.utils.hm3d_utils import mask_shapeconv_new_cats

from habitat_baselines.il.env_based.policy.semantic_predictor import SemanticPredictor

config = habitat.get_config("configs/tasks/objectnav_mp3d_il.yaml")


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def get_semantic_predictor(config):
    if config is None:
        return None
    device = torch.device("cuda", 0)
    semantic_predictor = SemanticPredictor(config.MODEL, device)
    semantic_predictor.eval()
    semantic_predictor.to(device)
    return semantic_predictor


def compute_goal_distance_to_view_points(sim, object_goals):
    avgs = []
    view_points_within_1m = []
    for goal in object_goals:
        goal_position = goal.position
        dists = []
        for view_point in goal.view_points:
            view_point_position = view_point.agent_state.position

            dist = sim.geodesic_distance(goal_position, view_point_position)
            # dist = np.linalg.norm(np.array(goal_position) - np.array(view_point_position), ord=2)
            if dist == np.inf:
                continue

            view_points_within_1m.append((view_point_position, dist < 1.0))
            dists.append(dist > 1.0)
        avgs.extend(dists)
    return avgs, view_points_within_1m


def find_closest_view_point(env, view_points_within_1m):
    closest_viewpoint = env.get_metrics()["distance_to_goal_v2"]["points"]
    
    is_within_1m = False
    for view_point, in_range in view_points_within_1m:
        if in_range:
            for point in closest_viewpoint:
                if np.allclose(view_point, point):
                    is_within_1m = True
                    break
            if is_within_1m:
                break
    return is_within_1m


def run_reference_replay(
    cfg,
    num_episodes=None,
    output_prefix=None,
    append_instruction=False,
    save_videos=False,
    save_step_image=False,
    config=None
):
    semantic_predictor = get_semantic_predictor(config)

    possible_actions = cfg.TASK.POSSIBLE_ACTIONS  
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        episode_meta = []
        dists = []
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            obs = env.reset()

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode

            for step_id, data in enumerate(env.current_episode.reference_replay[step_index:]):
                action = possible_actions.index(data.action)
                # action = data.action
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                obs = {"rgb": observations["rgb"]}
                masked_semantic = None
                if semantic_predictor is not None:
                    obs_semantic = semantic_predictor({"rgb": torch.tensor(observations["rgb"]).unsqueeze(0).cuda(), "depth": torch.tensor(observations["depth"]).unsqueeze(0).cuda()})
                    obs["semantic"] = obs_semantic[0].permute(1,2,0).long().cpu().numpy()
                    # obs["gt_semantic"] = mask_shapeconv_new_cats(obs["semantic"])
                    # obs["gt_semantic"] = obs_semantic[1].permute(1,2,0).long().cpu().numpy()

                info = env.get_metrics()
                frame = observations_to_image(obs, info)
                total_reward += info["simple_reward"]
                # is_in_range = find_closest_view_point(env, info["strict_success"]["reached_goal_within_1m"], info["strict_success"]["dtg"])

                print(info["simple_reward"], info["distance_to_goal"], info["strict_success"]["reached_goal_within_1m"], info["strict_success"]["dtg"])

                if append_instruction:
                    frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                if save_step_image:
                    save_image(frame, "trajectory_1/demo_{}_{}.png".format(ep_id, step_id))

                observation_list.append(frame)
                if action_name == "STOP":
                    break

            if save_videos:
                make_videos([observation_list], output_prefix, ep_id)
            print("Total reward: {}, Success: {}, Steps: {}, Attempts: {}".format(total_reward, info["success"], len(episode.reference_replay), episode.attempts))

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
        print("Avg distances: {}".format(sum(dists) / len(dists)))

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
        "--semantic-predictor-config", type=str, default=None
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.DATASET.MAX_EPISODE_STEPS = args.max_steps
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL", "TRAIN_SUCCESS",  "STRICT_SUCCESS", "SIMPLE_REWARD"] #, "TOP_DOWN_MAP"]
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS_WITHIN_1M = True
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD_V2 = False
    cfg.TASK.SIMPLE_REWARD.USE_DTG_REWARD = True
    cfg.freeze()

    model_config = None
    if not args.semantic_predictor_config is None:
        model_config = habitat.get_config(args.semantic_predictor_config)

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix,
        append_instruction=args.append_instruction,
        save_videos=args.save_videos,
        save_step_image=args.save_step_image,
        config=model_config,
    )


if __name__ == "__main__":
    main()
