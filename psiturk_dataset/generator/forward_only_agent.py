import argparse
import attr
import math
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
    im.save("demos/easy_scenes/" + file_name)


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


def calculate_turnn_angle(env, goal_position):
    agent_state = env._sim.get_agent(0).get_state()
    object_position = goal_position
    agent_to_obj = object_position - agent_state.position
    #agent_local_forward = np.array([0, 0, -1.0])
    # print(env._sim._default_agent.body.object.absolute_transformation())
    agent_local_forward = env._sim._default_agent.body.object.absolute_transformation().transform_vector(mn.Vector3(0, 0, -1))
    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
    flat_to_obj /= flat_dist_to_obj
    # move the agent closer to the objects if too far (this will be projected back to floor in set)
    was_gt = False
    if flat_dist_to_obj > 3.0:
        was_gt = True
        agent_state.position = object_position - flat_to_obj * 3.0
    # unit y normal plane for rotation
    det = (
        flat_to_obj[0] * agent_local_forward[2]
        - agent_local_forward[0] * flat_to_obj[2]
    )
    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
    stop = "no_stop"
    return np.rad2deg(turn_angle), stop, turn_angle


def get_closest_goal(episode, sim):
    goals = []
    distance = []
    min_dist = 1000.0
    goal_location = None
    agent_position = sim.get_agent_state().position
    goal_rotation = sim.get_agent_state().rotation
    for goal in episode.goals:
        for view_point in goal.view_points:
            position = view_point.agent_state.position
            rotation = view_point.agent_state.rotation
            
            dist = sim.geodesic_distance(
                agent_position, position
            )
            if min_dist > dist:
                min_dist = dist
                goal_location = position
                goal_rotation = rotation
    return goal_location, goal_rotation


def generate_trajectories(cfg, episode_path, output_prefix="s_path", enable_turns=False, output_path=""):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        goal_radius = 0.1
        spl = 0
        total_success = 0.0
        total_episodes = 0.0
        fixed_episodes = 0
        turn_fix_episodes = 0

        observation_list = []
        info = {}

        logger.info("Total episodes: {}".format(len(env.episodes)))
        success_episodes = []
        for ep_id in range(len(env.episodes)):
            env.reset()

            episode = env.current_episode
            goal_position, goal_rotation = get_closest_goal(episode, env.sim)

            turns_count = -1
            while not env.episode_over:
                best_action = HabitatSimActions.MOVE_FORWARD

                if enable_turns:
                    angle, stop, rad_agle = calculate_turnn_angle(env, goal_position)
                    if turns_count == -1:
                        turns_count = abs(angle) / 30 + 1
                    
                    if turns_count > 0:
                        if angle > 0:
                            best_action = HabitatSimActions.TURN_LEFT
                        else:
                            best_action = HabitatSimActions.TURN_RIGHT
                        turns_count -= 1

                if "distance_to_goal" in info.keys() and info["distance_to_goal"] < 0.1 and best_action != HabitatSimActions.STOP:
                    best_action = HabitatSimActions.STOP
                    fixed_episodes += 1

                observations = env.step(best_action)

                info = env.get_metrics()
                # Generate frames
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                save_image(frame, "ep_{}.jpg".format(ep_id))
                
                # frame = append_text_to_image(frame, "Find: {}".format(env.current_episode.object_category))
                # observation_list.append(frame)
                success = info["success"]
                break

            total_success += success
            spl += info["spl"]
            total_episodes += 1
            if success:
                success_episodes.append({
                    "episode_id": episode.episode_id,
                    "scene_id": episode.scene_id,
                    "info": info
                })

            logger.info("Episode success: {}, Total: {}, Success: {}, Fixed episodes: {}".format(success, total_episodes, total_success/total_episodes, fixed_episodes))
        
        print("Total episodes: {}".format(total_episodes))

        print("\n\nEpisode success: {}".format(total_success / total_episodes))
        print("Total episode success: {}".format(total_success))
        print("SPL: {}, {}, {}".format(spl/total_episodes, spl, total_episodes))
        print("Success: {}, {}, {}".format(total_success/total_episodes, total_success, total_episodes))
        write_json(success_episodes, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=str, default="data/episodes/sampled.json.gz"
    )
    parser.add_argument(
        "--enable-turns", dest="enable_turns", action="store_true"
    )
    parser.add_argument(
        "--output-path", type=str, default="success.json"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.episodes
    cfg.freeze()

    observations = generate_trajectories(cfg, args.episodes, enable_turns=args.enable_turns, output_path=args.output_path)

if __name__ == "__main__":
    main()
