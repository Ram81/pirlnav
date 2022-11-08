import argparse
import habitat
import os
import torch
import numpy as np

from PIL import Image
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from scripts.parsing.parse_objectnav_dataset import write_json

from habitat.utils.visualizations import maps
from scripts.utils.map_utils import draw_path, add_foreground
from scripts.utils.utils import load_json_dataset
from PIL import Image, ImageEnhance
from habitat.tasks.nav.nav import MAP_THICKNESS_SCALAR
from habitat import logger
from habitat_sim.utils.common import quat_to_coeffs

config = habitat.get_config("configs/tasks/objectnav_mp3d_il_video.yaml")


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def draw_point(tdm, point, color):

    point_padding = 10 * int(
        np.ceil(256 / MAP_THICKNESS_SCALAR)
    )
    tdm[
        point[0] - point_padding : point[0] + point_padding + 1,
        point[1] - point_padding : point[1] + point_padding + 1,
    ] = color
    return tdm


def get_foreground(sim, positions, episode):
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

    tdm = draw_point(tdm, points[0], maps.MAP_SOURCE_POINT_INDICATOR)
    tdm = draw_point(tdm, points[-1], maps.MAP_VIEW_POINT_INDICATOR)

    point_padding = 10 * int(
        np.ceil(256 / MAP_THICKNESS_SCALAR)
    )

    for goal in episode.goals:
        try:
            goal_point = maps.to_grid(goal.position[2], goal.position[0], tdm.shape[:2], sim)
            tdm = draw_point(
                tdm, goal_point, maps.MAP_TARGET_POINT_INDICATOR
            )
        except AttributeError:
            pass

    tdm = COLORS[tdm]
    foregrounds.append(tdm)
    return foregrounds


def overlay_top_down_map(positions, sim, scene_name, episode_id, baseline, episode, N=1):
    foregrounds = get_foreground(sim, positions, episode)
    background = Image.open("demos/original/{}.png".format(scene_name))

    enhancer = ImageEnhance.Contrast(background)
    background = enhancer.enhance(0.4)
    enhancer = ImageEnhance.Brightness(background)
    background = enhancer.enhance(1.6)

    foreground = foregrounds[0]
    background = add_foreground(background, foreground, 60, 30)

    dir_name = "demos/overlayed/{}".format(baseline)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    path = "{}/{}.png".format(dir_name, episode_id)
    background.save(path)


def episode_to_evaluation_meta(evaluation_meta):
    episode_eval_meta_map = {}
    for episode_meta in evaluation_meta:
        episode_eval_meta_map[episode_meta["episode_id"]] = episode_meta
    return episode_eval_meta_map


def run_reference_replay(
    cfg,
    num_episodes=None,
    output_prefix=None,
    append_instruction=False,
    save_videos=False,
    save_step_image=False,
    separate_top_down_map=False,
    save_top_down_map=False,
    evaluation_meta_path=None,
    baseline="test",
    specific_episode_id=None
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS

    evaluation_meta = None
    if evaluation_meta_path is not None:
        evaluation_meta = load_json_dataset(evaluation_meta_path)
        episode_eval_meta_map = episode_to_evaluation_meta(evaluation_meta)

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
            hab_on_web_trajectory = []

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode
            if specific_episode_id is not None and episode.episode_id != specific_episode_id:
                continue
            positions = [env._sim.get_agent_state()]

            reference_replay = env.current_episode.reference_replay
            if evaluation_meta is not None:
                reference_replay = episode_eval_meta_map[env.current_episode.episode_id]["actions"]
                if reference_replay[0] != "STOP":
                    step_index = 0
                print("Expected succesS: {}".format(episode_eval_meta_map[env.current_episode.episode_id]["metrics"]["success"]))


            for step_id, data in enumerate(reference_replay[step_index:]):
                if type(data) is str:
                    action = possible_actions.index(data)
                else:
                    action = possible_actions.index(data.action)
                # action = data.action
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                obs = {"rgb": observations["rgb"]}

                info = env.get_metrics()
                agent_state = env._sim.get_agent_state()
                positions.append(agent_state)

                top_down_frame = None
                if not separate_top_down_map:
                    frame = observations_to_image(obs, info)
                else:
                    frame = observations_to_image(obs, {})
                    top_down_frame = observations_to_image(obs, info, top_down_map_only=True)

                if append_instruction:
                    frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                if save_step_image:
                    save_image(frame, "trajectory_1/demo_{}_{}.png".format(ep_id, step_id))

                observation_list.append(frame)
                hab_on_web_trajectory.append({
                    "position": np.array(agent_state.position).tolist(),
                    "rotation": quat_to_coeffs(agent_state.rotation).tolist(),
                    "sensor_data": {}
                })

                if separate_top_down_map:
                    top_down_map_observation_list.append(top_down_frame)

                if action_name == "STOP":
                    break

            if save_videos:
                make_videos([observation_list], output_prefix, ep_id)

            if save_top_down_map:
                make_videos([top_down_map_observation_list], "{}_top_down_map".format(output_prefix), ep_id)

            print("Total reward: {}, Success: {}, Steps: {}, Attempts: {}".format(total_reward, info["success"], len(reference_replay), episode.attempts))
            if "top_down_map" in info:
                del info["top_down_map"]
                del info["behavior_metrics"]

            episode_meta.append({
                "scene_id": episode.scene_id,
                "episode_id": episode.episode_id,
                "metrics": info,
                "attempts": episode.attempts,
                "object_category": episode.object_category
            })

            write_json({
                "trajectory": hab_on_web_trajectory
            }, "demos/overlayed/{}/demo_trajectory.json".format(baseline))

            overlay_top_down_map(positions, env._sim, episode.scene_id.split("/")[-2], episode.episode_id, baseline, episode)
            logger.info("Episodes done: {}/{}".format(ep_id, num_episodes))

        logger.info("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        logger.info("Success: {}, {}, {}".format(total_success/num_episodes, total_success, num_episodes))

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
        "--separate-top-down-map", action="store_true", dest="separate_top_down_map"
    )
    parser.add_argument(
        "--save-top-down-map", action="store_true", dest="save_top_down_map"
    )
    parser.add_argument(
        "--evaluation-meta-path", type=str, default=None
    )
    parser.add_argument(
        "--baseline", type=str, default="demo"
    )
    parser.add_argument(
        "--specific-episode-id", type=str, default="ziup5kvtCCR_26_[5.47169, 0.02122, 2.32604]_plant.png"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.DATASET.MAX_EPISODE_STEPS = args.max_steps
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    cfg.TASK.SENSORS = ['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"]
    cfg.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS_WITHIN_1M = False
    cfg.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD_V2 = False
    cfg.TASK.SIMPLE_REWARD.USE_DTG_REWARD = False
    cfg.TASK.SIMPLE_REWARD.USE_ANGLE_SUCCESS_REWARD = True
    cfg.DATASET.TYPE = "ObjectNav-v1"
    cfg.DATASET.CONTENT_SCENES = ["svBbv1Pavdk", "Dd4bFSTQ8gi", "QaLdnwvtxbs", "ziup5kvtCCR", "5cdEh9F2hJL", "mv2HUxq3B53", "Nfvxx8J5NCo"]
    cfg.DATASET.CONTENT_SCENES = ["ziup5kvtCCR"]
    cfg.freeze()

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix,
        append_instruction=args.append_instruction,
        save_videos=args.save_videos,
        save_step_image=args.save_step_image,
        separate_top_down_map=args.separate_top_down_map,
        save_top_down_map=args.save_top_down_map,
        evaluation_meta_path=args.evaluation_meta_path,
        baseline=args.baseline,
        specific_episode_id=args.specific_episode_id
    )


if __name__ == "__main__":
    main()

