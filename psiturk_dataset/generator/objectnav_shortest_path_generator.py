import argparse
import attr
import cv2
import habitat
import copy
import numpy as np
import magnum as mn
import sys

from habitat import Config, logger, get_config as get_task_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import make_video_cv2, observations_to_image, images_to_video, append_text_to_image

from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import quat_to_coeffs, quat_from_magnum, quat_from_angle_axis

from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset
from scipy.spatial.transform import Rotation
from habitat.utils.visualizations import maps

from PIL import Image

config = habitat.get_config("configs/tasks/shortest_path_objectnav_mp3d.yaml")

objectnav_scene_splits = {
    "split_1": ['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '5LpN3gDmAk7', '5q7pvUzZiYa', '759xd9YjKW5', '7y3sRwLe3Va', '82sE5b5pLXE', '8WUmhLawc2A', 'B6ByNegPMKs', 'D7G3Y4RVNrH', 'D7N2EKCX4Sj', 'E9uDoFAP3SH'],
    "split_2": ['EDJbREhghzL', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'JF19kD82Mey', 'JeFG25nYj2p', 'PX4nDJXEHrG', 'Pm6F8kyY3z2', 'PuKPg4mmafe', 'S9hNv5qa7GM', 'ULsKaCPVFJR', 'Uxmj2M2itWa', 'V2XKFyX4ASd', 'VFuaQ6m2Qom', 'VLzqgDo317F'],
    "split_3": ['VVfe2KiqLaN', 'Vvot9Ly1tCj', 'XcA2TqTSSAj', 'YmJkqBEsHnH', 'ZMojNkEp431', 'aayBHfsNo7d', 'ac26ZMwG7aT', 'b8cTxDM8gDG', 'cV4RVeZvu5T', 'dhjEzFoUFzH', 'e9zR4mvMWw7', 'gZ6f7yhEvPG', 'i5noydFURQK', 'jh4fc5c5qoQ'],
    "split_4": ['kEZ7cmS4wCh', 'mJXqzFtmKg4', 'p5wJjkQkbXX', 'pRbA3pwrgk9', 'qoiz87JEwZ2', 'r1Q1Z4BcV1o', 'r47D5H71a5s', 'rPc6DW4iMge', 's8pcmisQ38h', 'sKLMLpTHeUy', 'sT4fr6TAbpF', 'uNb9QFRL6hY', 'ur6pFq6Qu1A', 'vyrNrziPKCB'],
    "split_5": ["17DRP5sb8fy"]
}

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/s_path/" + file_name)


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos/s_path/", video_name=prefix)


def get_action(action):
    if action == HabitatSimActions.TURN_RIGHT:
        return "TURN_RIGHT"
    elif action == HabitatSimActions.TURN_LEFT:
        return "TURN_LEFT"
    elif action == HabitatSimActions.MOVE_FORWARD:
        return "MOVE_FORWARD"
    elif action == HabitatSimActions.LOOK_UP:
        return "LOOK_UP"
    elif action == HabitatSimActions.LOOK_DOWN:
        return "LOOK_DOWN"
    return "STOP"


def object_state_to_json(object_states):
    object_states_json = []
    for object_ in object_states:
        object_states_json.append(attr.asdict(object_))
    return object_states_json


def get_agent_pose(sim):
    agent_translation = sim._default_agent.body.object.translation
    agent_rotation = sim._default_agent.body.object.rotation
    sensor_data = {}
    for sensor_key, v in sim._default_agent._sensors.items():
        rotation = quat_from_magnum(v.object.rotation)
        rotation = quat_to_coeffs(rotation).tolist()
        translation = v.object.translation
        sensor_data[sensor_key] = {
            "rotation": rotation,
            "translation": np.array(translation).tolist()
        }
    
    return {
        "position": np.array(agent_translation).tolist(),
        "rotation": quat_to_coeffs(quat_from_magnum(agent_rotation)).tolist(),
        "sensor_data": sensor_data
    }


def get_action_data(action, sim):
    data = {}
    data["action"] = get_action(action)
    data["agent_state"] = get_agent_pose(sim)
    return data


def get_episode_json(episode, reference_replay):
    episode.reference_replay = reference_replay
    #episode._shortest_path_cache = None
    # episode.goals = []
    episode.scene_id = episode.scene_id
    ep_json = attr.asdict(episode)
    del ep_json["_shortest_path_cache"]
    return ep_json


def is_object_gripped(sim):
    return sim._prev_step_data["gripped_object_id"] != -1


def execute_grab(sim, prev_action):
    if sim._prev_sim_obs["object_under_cross_hair"] !=  -1 and prev_action != HabitatSimActions.GRAB_RELEASE:
        return True
    return False


def is_prev_action_look_down(action):
    return action == HabitatSimActions.LOOK_DOWN


def is_prev_action_turn_left(action):
    return action == HabitatSimActions.TURN_LEFT


def is_prev_action_turn_right(action):
    return action == HabitatSimActions.TURN_RIGHT


def get_closest_goal(episode, sim, follower):
    goals = []
    distance = []
    min_dist = 1000.0
    goal_location = None
    agent_position = sim.get_agent_state().position
    for goal in episode.goals:
        for view_point in goal.view_points:
            position = view_point.agent_state.position
            
            dist = sim.geodesic_distance(
                agent_position, position
            )
            if min_dist > dist:
                min_dist = dist
                goal_location = position
    return goal_location


def create_episode_trajectory(trajectory, episode):
    goal_positions = []
    for goal in episode.goals:
        for view_point in goal.view_points:
            goal_positions.append(view_point.agent_state.position)
    data = {
        "trajectory": trajectory,
        "object_goal": episode.object_category,
        "goal_locations": goal_positions,
        "scene_id": episode.scene_id.split("/")[-1]
    }
    return data

def get_coverage(info):
    top_down_map = info["map"]
    visted_points = np.where(top_down_map <= 9, 0, 1)
    coverage = np.sum(visted_points) / get_navigable_area(info)
    return coverage


def get_navigable_area(info):
    top_down_map = info["map"]
    navigable_area = np.where(((top_down_map == 1) | (top_down_map >= 10)), 1, 0)
    return np.sum(navigable_area)


def get_visible_area(info):
    fog_of_war_mask = info["fog_of_war_mask"]
    visible_area = fog_of_war_mask.sum() / get_navigable_area(info)
    return visible_area


def save_top_down_map(info):
    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], 512
    )
    save_image(top_down_map, "top_down_map.png")


def generate_trajectories(cfg, episode_path, output_prefix="s_path", scene_id="data", output_path=""):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        goal_radius = 0.1
        spl = 0
        total_success = 0.0
        total_episodes = 0.0
        total_coverage = 0.0
        visible_area = 0.0
        fixed_episodes = 0

        dataset = load_dataset(episode_path)

        dataset["episodes"] = []
        failed_dataset = copy.deepcopy(dataset)
        failed_dataset["episodes"] = []

        logger.info("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            follower = ShortestPathFollower(
                env._sim, goal_radius, False
            )
            observation_list = []
            env.reset()
            goal_idx = 0
            success = 0
            reference_replay = []
            episode = env.current_episode
            goal_position = get_closest_goal(episode, env.sim, follower)
            info = {}
            replay_data = []
            reference_replay.append(get_action_data(HabitatSimActions.STOP, env._sim))
            i = 0
            if goal_position is None:
                continue
            while not env.episode_over:
                best_action = follower.get_next_action(
                    goal_position
                )

                if "distance_to_goal" in info.keys() and info["distance_to_goal"] < 0.1 and best_action != HabitatSimActions.STOP:
                    best_action = HabitatSimActions.STOP
                    fixed_episodes += 1

                observations = env.step(best_action)

                info = env.get_metrics()
                # Generate frames
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                frame = append_text_to_image(frame, "Find: {}".format(env.current_episode.object_category))
                observation_list.append(frame)
                success = info["success"]

                action_data = get_action_data(best_action, env._sim)
                reference_replay.append(action_data)
                replay_data.append(get_agent_pose(env._sim))
                i += 1

            ep_data = get_episode_json(env.current_episode, reference_replay)
            del ep_data["_shortest_path_cache"]
            total_success += success
            spl += info["spl"]
            total_episodes += 1

            #visible_area += get_visible_area(info["top_down_map"])
            # total_coverage += get_coverage(info["top_down_map"])

            logger.info("Episode success: {}, Total: {}, Success: {}, Fixed episodes: {}, Steps: {}".format(success, total_episodes, total_success/total_episodes, fixed_episodes, i))
            # if not success:
            #     make_videos([observation_list], output_prefix, ep_id)
            # save_image(frame, "{}_s_path.png".format(ep_data["episode_id"]))
            if success:
                dataset["episodes"].append(ep_data)
            if not success:
                failed_dataset["episodes"].append(ep_data)
            
            # trajectory = create_episode_trajectory(replay_data, episode)
            # write_json(trajectory, "data/visualizations/teaser/trajectories/{}_{}_s_path_trajectory.json".format(episode.scene_id.split("/")[-1].split(".")[0], ep_id))
            # save_top_down_map(info)

            if (ep_id + 1) % 1000 == 0:
                logger.info("Saving checkpoint at {} episodes".format(ep_id))
                write_json(dataset, "{}/{}.json".format(output_path, scene_id))
                write_gzip("{}/{}.json".format(output_path, scene_id), "{}/{}.json".format(output_path, scene_id))
        
        print("Total episodes: {}".format(total_episodes))

        print("\n\nEpisode success: {}".format(total_success / total_episodes))
        print("Total episode success: {}".format(total_success))
        print("SPL: {}, {}, {}".format(spl/total_episodes, spl, total_episodes))
        print("SPL: {}, {}, {}".format(spl/total_success, spl, total_success))
        print("Success: {}, {}, {}".format(total_success/total_episodes, total_success, total_episodes))
        print("Coverage: {}, {}, {}".format(total_coverage/total_episodes, total_coverage, total_episodes))
        print("Visible area: {}, {}, {}".format(visible_area/total_episodes, visible_area, total_episodes))
        print("Total sample episodes: {}/{}".format(len(dataset["episodes"]), total_episodes))
        write_json(dataset, "{}/{}.json".format(output_path, scene_id))
        write_gzip("{}/{}.json".format(output_path, scene_id), "{}/{}.json".format(output_path, scene_id))

        if len(failed_dataset["episodes"]) > 0:
            write_json(failed_dataset, "{}/{}_failed.json".format(output_path, scene_id))
            write_gzip("{}/{}_failed.json".format(output_path, scene_id), "{}/{}_failed.json".format(output_path, scene_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes", type=int, default=1
    )
    parser.add_argument(
        "--scene", type=str, default="objectnav_round_2"
    )
    parser.add_argument(
        "--split", type=str, default="split_1"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/episodes/sampled.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/s_path_objectnav/round_2_full/extended_v2/"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.episodes
    split = objectnav_scene_splits[args.split]
    if len(split) > 1:
        cfg.DATASET.CONTENT_SCENES = split
    cfg.freeze()

    observations = generate_trajectories(cfg, args.episodes, scene_id=args.scene, output_path=args.output_path)

if __name__ == "__main__":
    main()
