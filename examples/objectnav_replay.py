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
from habitat.utils.visualizations import maps
from scripts.utils.map_utils import draw_path, add_foreground
from PIL import Image, ImageEnhance

config = habitat.get_config("configs/tasks/objectnav_mp3d_il_video.yaml")


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


def get_foreground(sim, positions):
    original_tdm = maps.get_topdown_map_from_sim(sim, meters_per_pixel=0.01)

    ALPHA = 180

    COLORS = np.copy(maps.TOP_DOWN_MAP_COLORS)
    COLORS = np.concatenate([COLORS, 255*np.ones_like(COLORS)[:, :1]], axis=1)

    COLORS[0] = [255, 255, 255, 0]

    COLORS[10] = [112, 173, 71, 255]
    COLORS[11] = [112, 173, 71, ALPHA]

    COLORS[12] = [91, 155, 213, 255]
    COLORS[13] = [91, 155, 213, ALPHA]

    COLORS[14] = [237, 125, 49, 255]
    COLORS[15] = [237, 125, 49, ALPHA]


    COLORS[16] = [255, 192, 0, 255]
    COLORS[17] = [255, 192, 0, ALPHA]


    color = 10
    foregrounds = []
    tdm = np.zeros_like(original_tdm)
    points = [maps.to_grid(p.position[2], p.position[0], tdm.shape[:2], sim) for p in positions]
    draw_path(tdm, points, color=color, thickness=9)

    tdm = COLORS[tdm]
    foregrounds.append(tdm)
    return foregrounds


def overlay_top_down_map(positions, sim, scene_name, N=1):
    foregrounds = get_foreground(sim, positions)
    background = Image.open("demos/00853-5cdEh9F2hJL.png")

    enhancer = ImageEnhance.Contrast(background)
    background = enhancer.enhance(0.4)
    enhancer = ImageEnhance.Brightness(background)
    background = enhancer.enhance(1.6)

    foreground = foregrounds[0]
    background = add_foreground(background, foreground, 60, 30)

    path = "demos/overlayed/{}_overlay.png".format(scene_name)
    background.save(path)

    #save_image(foregrounds[0], "original/{}.png".format(scene_name))


def run_reference_replay(
    cfg,
    num_episodes=None,
    output_prefix=None,
    append_instruction=False,
    save_videos=False,
    save_step_image=False,
    config=None,
    separate_top_down_map=False,
    save_top_down_map=False,
):
    semantic_predictor = get_semantic_predictor(config)

    possible_actions = cfg.TASK.POSSIBLE_ACTIONS  
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        episode_meta = []
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            top_down_map_observation_list = []
            obs = env.reset()

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode
            positions = [env._sim.get_agent_state()]
            print(episode.episode_id, episode.object_category)

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

                info = env.get_metrics()
                positions.append(env._sim.get_agent_state())

                top_down_frame = None
                if not separate_top_down_map:
                    frame = observations_to_image(obs, info)
                else:
                    frame = observations_to_image(obs, {})
                    top_down_frame = observations_to_image(obs, info, top_down_map_only=True)

                # total_reward += info["simple_reward"]

                if append_instruction:
                    frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                if save_step_image:
                    save_image(frame, "trajectory_1/demo_{}_{}.png".format(ep_id, step_id))

                observation_list.append(frame)

                if separate_top_down_map:
                    top_down_map_observation_list.append(top_down_frame)
                
                print(step_index + step_id, action, observations["inflection_weight"])

                if action_name == "STOP":
                    break

            if save_videos:
                make_videos([observation_list], output_prefix, ep_id)

            if save_top_down_map:
                make_videos([top_down_map_observation_list], "{}_top_down_map".format(output_prefix), ep_id)

            print("Total reward: {}, Success: {}, Steps: {}, Attempts: {}".format(total_reward, info["success"], len(episode.reference_replay), episode.attempts))
            # del info["top_down_map"]
            # del info["behavior_metrics"]

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

            # overlay_top_down_map(positions, env._sim, episode.scene_id.split("/")[-1])
            # save_image(top_down_map_observation_list[-1], "original/{}.png".format(episode.scene_id.split("/")[-1]))

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
    parser.add_argument(
        "--separate-top-down-map", action="store_true", dest="separate_top_down_map"
    )
    parser.add_argument(
        "--save-top-down-map", action="store_true", dest="save_top_down_map"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.DATASET.MAX_EPISODE_STEPS = args.max_steps
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    #cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL", "TRAIN_SUCCESS",  "STRICT_SUCCESS", "ANGLE_TO_GOAL", "ANGLE_SUCCESS", "SIMPLE_REWARD", "TOP_DOWN_MAP", "BEHAVIOR_METRICS"]
    cfg.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = False
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS_WITHIN_1M = False
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = False
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD_V2 = False
    cfg.TASK.SIMPLE_REWARD.USE_DTG_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_ANGLE_SUCCESS_REWARD = True
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
        separate_top_down_map=args.separate_top_down_map,
        save_top_down_map=args.save_top_down_map,
    )


if __name__ == "__main__":
    main()

