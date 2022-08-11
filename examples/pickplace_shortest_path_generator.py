import argparse
import attr
import cv2
import habitat
import json
import math
import sys
import time
import os
import numpy as np
import magnum as mn

from habitat import Config, get_config as get_task_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import make_video_cv2, observations_to_image, images_to_video, append_text_to_image

from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import quat_to_coeffs, quat_from_magnum, quat_from_angle_axis

from psiturk_dataset.utils.utils import write_json, write_gzip
from scipy.spatial.transform import Rotation

from PIL import Image

config = habitat.get_config("configs/tasks/rearrangement_shortest_trajectory.yaml")

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def get_action(action):
    if action == HabitatSimActions.TURN_RIGHT:
        return "TURN_RIGHT"
    elif action == HabitatSimActions.TURN_LEFT:
        return "TURN_LEFT"
    elif action == HabitatSimActions.MOVE_FORWARD:
        return "MOVE_FORWARD"
    elif action == HabitatSimActions.MOVE_BACKWARD:
        return "MOVE_BACKWARD"
    elif action == HabitatSimActions.LOOK_UP:
        return "LOOK_UP"
    elif action == HabitatSimActions.LOOK_DOWN:
        return "LOOK_DOWN"
    elif action == HabitatSimActions.GRAB_RELEASE:
        return "GRAB_RELEASE"
    elif action == HabitatSimActions.NO_OP:
        return "NO_OP"
    return "STOP"



def object_state_to_json(object_states):
    object_states_json = []
    for object_ in object_states:
        object_states_json.append(attr.asdict(object_))
    return object_states_json


def get_action_data(action, sim):
    data = {}
    if action == HabitatSimActions.GRAB_RELEASE:
        data["gripped_object_id"] = sim.gripped_object_id
        data["is_grab_action"] = sim._prev_step_data["is_grab_action"]
        data["is_release_action"] = sim._prev_step_data["is_release_action"]
        action_data = {}
        if data["is_release_action"]:
            action_data["new_object_translation"] = sim._prev_step_data["new_object_translation"]
            action_data["new_object_id"] = sim._prev_step_data["new_object_id"]
            action_data["object_handle"] = sim._prev_step_data["object_handle"]
            action_data["gripped_object_id"] = sim._prev_step_data["gripped_object_id"]
        else:
            action_data["gripped_object_id"] = sim._prev_step_data["gripped_object_id"]
        data["action_data"] = action_data
        data["collision"] = False
    else:
        data["collision"] = sim._prev_step_data["collided"]
        data["object_under_cross_hair"] = sim._prev_step_data["nearest_object_id"]
        data["nearest_object_id"] = sim._prev_step_data["nearest_object_id"]
        data["gripped_object_id"] = sim._prev_step_data["gripped_object_id"]
    data["action"] = get_action(action)
    data["agent_state"] = sim.get_agent_pose()
    data["object_states"] = object_state_to_json(sim.get_current_object_states())
    return data


def get_episode_json(episode, reference_replay):
    episode.reference_replay = reference_replay
    episode._shortest_path_cache = None
    episode.scene_id = episode.scene_id.split("/")[-1]
    return attr.asdict(episode)


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


def get_next_action(prev_actions, num_steps, turn_stack):
    if num_steps < 10:
        return HabitatSimActions.LOOK_DOWN
    else:
        if prev_actions[-1] == HabitatSimActions.LOOK_DOWN:
            return HabitatSimActions.TURN_LEFT
        elif prev_actions[-2] == HabitatSimActions.LOOK_DOWN and np.sum(turn_stack) == -1:
            return HabitatSimActions.TURN_LEFT
        elif prev_actions[-1] == HabitatSimActions.TURN_LEFT and np.sum(turn_stack) >= -3 and np.sum(turn_stack) < 0:
            return HabitatSimActions.TURN_LEFT
        elif np.sum(turn_stack) < 0:
            return HabitatSimActions.TURN_RIGHT
        elif prev_actions[-1] == HabitatSimActions.TURN_RIGHT and np.sum(turn_stack) <= 0:
            return HabitatSimActions.TURN_RIGHT
        elif prev_actions[-1] == HabitatSimActions.TURN_RIGHT and np.sum(turn_stack) <= 4:
            return HabitatSimActions.TURN_RIGHT
        elif np.sum(turn_stack) > 0:
            return HabitatSimActions.TURN_LEFT
        return HabitatSimActions.LOOK_DOWN


def get_next_action_after_pick_up(prev_actions, num_steps, turn_stack):
    if num_steps < 10:
        return HabitatSimActions.LOOK_DOWN
    else:
        if prev_actions[-1] == HabitatSimActions.LOOK_DOWN:
            return HabitatSimActions.TURN_LEFT
        elif prev_actions[-2] == HabitatSimActions.LOOK_DOWN and np.sum(turn_stack) == -1:
            return HabitatSimActions.TURN_LEFT
        elif prev_actions[-1] == HabitatSimActions.TURN_LEFT and np.sum(turn_stack) >= -3 and np.sum(turn_stack) < 0:
            return HabitatSimActions.TURN_LEFT
        elif np.sum(turn_stack) < 0:
            return HabitatSimActions.TURN_RIGHT
        elif prev_actions[-1] == HabitatSimActions.TURN_RIGHT and np.sum(turn_stack) <= 0:
            return HabitatSimActions.TURN_RIGHT
        elif prev_actions[-1] == HabitatSimActions.TURN_RIGHT and np.sum(turn_stack) <= 4:
            return HabitatSimActions.TURN_RIGHT
        elif np.sum(turn_stack) > 0:
            return HabitatSimActions.TURN_LEFT
        return HabitatSimActions.LOOK_DOWN


def calculate_turn_angle(agent, target, fwd, fwd_2):    
    # qd = agent.conjugated() * target

    # theta = math.asin( 2 * (qd.scalar * qd.vector.y - qd.vector.z * qd.vector.x) )
    fw_agent = agent.transform_vector(mn.Vector3(0, 0, -1))
    fw_target = target.transform_vector(mn.Vector3(0, 0, -1))

    angle_a = math.atan2(fw_agent.x, fw_agent.z)
    angle_b = math.atan2(fw_target.x, fw_target.z)
    angle_a = np.rad2deg(angle_a)
    angle_b = np.rad2deg(angle_b)
    theta = angle_b - angle_a
    # theta = (theta + 180) % 360 - 180

    # rot = Rotation.from_quat(quat_to_coeffs(quat_from_magnum(qd)))
    # theta = rot.as_euler('xyz', degrees=True)[1]
    # if flg == 0 or flg is None:
    #     print(rot.as_euler('xyz', degrees=True), math.degrees(theta), theta)
    
    return theta


def calculate_turnn_angle(env, goal_idx, best_action):
    agent_state = env._sim.get_agent(0).get_state()
    object_position = env._sim.get_translation(goal_idx)
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
    # if goal_idx == 1 and (best_action == 0 or best_action is None):
    #     agent_state.rotation = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))
    #     # need to move the sensors too
    #     for sensor in agent_state.sensor_states:
    #         agent_state.sensor_states[sensor].rotation = agent_state.rotation
    #         agent_state.sensor_states[
    #             sensor
    #         ].position = agent_state.position + np.array([0, 1.5, 0])
    #     env._sim.get_agent(0).set_state(agent_state)
    #     stop = "stop"
    #     print("was gt: {}".format(was_gt))
    return np.rad2deg(turn_angle), stop, turn_angle


def generate_trajectories(cfg, num_episodes=1, output_prefix="s_path", scene_id="", output_path=""):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        goal_radius = 0.9

        total_success = 0.0
        total_episodes = 0.0

        dataset = {
            "episodes": [],
            "instruction_vocab": {
                "sentences": []
            }
        }
        failed_dataset = {
            "episodes": [],
            "instruction_vocab": {
                "sentences": []
            }
        }

        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            follower = ShortestPathFollower(
                env._sim, goal_radius, False
            )
            observation_list = []
            obs = env.reset()
            goal_idx = 0
            prev_action = 0
            num_steps = 0
            turn_stack = []
            success = 0
            near_object = False
            near_receptacle = False
            is_gripped = False
            is_released = False
            stack_cleared = False
            reference_replay = []
            prev_actions = []
            turn_step_indices = [-1, -1]
            num_look_down = 0
            after_release_steps = 0

            turns_done = False
            turns_count = 0
            action_data = get_action_data(HabitatSimActions.STOP, env._sim)
            reference_replay.append(action_data)
            while not env.episode_over:
                best_action = follower.get_next_action(
                    env.current_episode.goals[goal_idx].position
                )

                angle, stop, rad_agle = calculate_turnn_angle(env, goal_idx, best_action)

                if (best_action is None or best_action == 0) and not turns_done and turns_count == 0:
                    turns_count = abs(angle) / 5 + 1

                if turns_count == 0 and (best_action is None or best_action == 0) and not turns_done:
                    turns_done = True

                if (best_action is None or best_action == 0) and turns_count > 0 and not turns_done:
                    if angle > 0:
                        best_action = HabitatSimActions.TURN_LEFT
                    else:
                        best_action = HabitatSimActions.TURN_RIGHT
                    turns_count -= 1

                if (best_action is None or best_action == 0):
                    if goal_idx == 0:
                        near_object = True
                        stack_cleared = False
                        best_action = get_next_action(prev_actions, num_steps, turn_stack)
                        if best_action == HabitatSimActions.TURN_LEFT:
                            turn_stack.append(-1)
                        elif best_action == HabitatSimActions.TURN_RIGHT:
                            turn_stack.append(1)
                        elif best_action == HabitatSimActions.LOOK_DOWN:
                            turn_stack = []
                            turn_step_indices[1] = len(reference_replay) - 1
                            stack_cleared = True
                            num_look_down += 1
                        if len(turn_stack) == 1:
                            turn_step_indices[0] = len(reference_replay)
                        num_steps += 1
                    elif goal_idx == 1 and after_release_steps < 5:
                        near_receptacle = True
                        stack_cleared = False
                        best_action = get_next_action_after_pick_up(prev_actions, num_steps, turn_stack)
                        if best_action == HabitatSimActions.TURN_LEFT:
                            turn_stack.append(-1)
                        elif best_action == HabitatSimActions.TURN_RIGHT:
                            turn_stack.append(1)
                        elif best_action == HabitatSimActions.LOOK_DOWN:
                            turn_stack = []
                            turn_step_indices[1] = len(reference_replay) - 1
                            stack_cleared = True
                        if len(turn_stack) == 1:
                            turn_step_indices[0] = len(reference_replay)

                        if is_released:
                            after_release_steps += 1
                        num_steps += 1
                
                if is_gripped and not is_released and num_look_down > 0 and goal_idx == 1:
                    stack_cleared = False
                    best_action = HabitatSimActions.LOOK_UP
                    num_look_down -= 1

                if execute_grab(env._sim, prev_action) and not is_gripped and near_object:
                    best_action = HabitatSimActions.GRAB_RELEASE
                
                if near_receptacle and is_gripped and not is_released and execute_grab(env._sim, prev_action):
                    best_action = HabitatSimActions.GRAB_RELEASE
                    is_released = True

                if success:
                    best_action = HabitatSimActions.STOP

                observations = env.step(best_action)
                if near_receptacle and is_gripped and not is_released and execute_grab(env._sim, prev_action) and not is_object_gripped(env._sim):
                    is_released = True
                # Switch to searching for receptacle
                if is_object_gripped(env._sim) and best_action == HabitatSimActions.GRAB_RELEASE:
                    goal_idx = 1
                    is_gripped = True
                    num_steps = 0
                    turn_step_indices = [-1, -1]
                    turns_done = False
                    turns_count = 0

                if stack_cleared:
                    if turn_step_indices[0] > 0:
                        reference_replay = reference_replay[:turn_step_indices[0]]
                    turn_step_indices = [-1, -1]

                info = env.get_metrics()
                # Generate frames
                frame = observations_to_image({"rgb": observations["rgb"], "depth": observations["depth"]}, {})
                frame = append_text_to_image(frame, "Instruction: {}".format(env.current_episode.instruction.instruction_text))
                observation_list.append(frame)
                success += info["success"]

                prev_action = best_action
                prev_actions.append(best_action)

                action_data = get_action_data(best_action, env._sim)
                reference_replay.append(action_data)

            ep_data = get_episode_json(env.current_episode, reference_replay)
            del ep_data["_shortest_path_cache"]
            print(info)
            print(prev_action, success, success != 0 and prev_action == HabitatSimActions.STOP)
            print("Episode success: {}, Total episodes: {}, Total success: {}".format(success, total_episodes, total_success))
            total_success += success
            total_episodes += 1
            # if not success:
            #     make_videos([observation_list], output_prefix, ep_id)

            if total_episodes % 500 == 0:
                print("Checkpointing at {} episodes".format(total_episodes))
                write_json(dataset, "{}/{}.json".format(output_path, scene_id))
                write_gzip("{}/{}.json".format(output_path, scene_id), "{}/{}.json".format(output_path, scene_id))


            if success:
                dataset["episodes"].append(ep_data)
            if not success:
                failed_dataset["episodes"].append(ep_data)

        print("\n\nEpisode success: {}".format(total_success / total_episodes))
        print("Total sample episodes: {}".format(len(dataset["episodes"])))
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
        "--scene", type=str, default="empty_house"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/episodes/sampled.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sampled.json.gz"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.CONTENT_SCENES = [args.scene]
    cfg.DATASET.DATA_PATH = args.episodes
    cfg.freeze()

    observations = generate_trajectories(cfg, args.num_episodes, scene_id=args.scene, output_path=args.output_path)

if __name__ == "__main__":
    main()
