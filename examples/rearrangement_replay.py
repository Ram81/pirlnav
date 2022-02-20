import argparse
import cv2
import habitat
import json
import sys
import time
import os
import numpy as np

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from habitat.utils.visualizations.utils import make_video_cv2, observations_to_image, images_to_video, append_text_to_image

from threading import Thread
from time import sleep

from PIL import Image

config = habitat.get_config("configs/tasks/rearrangement_video.yaml")


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def save_grab_release_frames(env, i):
    if i + 1 <= len(env.current_episode.reference_replay) - 1:
        prev_action = env.current_episode.reference_replay[i - 1]
        prev_prev_action = env.current_episode.reference_replay[i - 2]
        if data.action == "grabReleaseObject":
            im = Image.fromarray(observations["rgb"])
            im.save('outfile{}.jpg'.format(i))
        if prev_action.action == "grabReleaseObject" and data.action == "stepPhysics":
            im = Image.fromarray(observations["rgb"])
            im.save('outfile{}.jpg'.format(i))
        if prev_prev_action.action == "grabReleaseObject" and prev_action.action == "stepPhysics":
            im = Image.fromarray(observations["rgb"])
            im.save('outfile{}.jpg'.format(i))


def make_videos(observations_list, output_prefix, ep_id):
    #print(observations_list[0][0].keys(), type(observations_list[0][0]))
    prefix = output_prefix + "_{}".format(ep_id)
    # make_video_cv2(observations_list[0], prefix=prefix, open_vid=False)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def get_habitat_sim_action(data):
    if data.action == "turnRight":
        return HabitatSimActions.TURN_RIGHT
    elif data.action == "turnLeft":
        return HabitatSimActions.TURN_LEFT
    elif data.action == "moveForward":
        return HabitatSimActions.MOVE_FORWARD
    elif data.action == "moveBackward":
        return HabitatSimActions.MOVE_BACKWARD
    elif data.action == "lookUp":
        return HabitatSimActions.LOOK_UP
    elif data.action == "lookDown":
        return HabitatSimActions.LOOK_DOWN
    elif data.action == "grabReleaseObject":
        return HabitatSimActions.GRAB_RELEASE
    elif data.action == "stepPhysics":
        return HabitatSimActions.NO_OP
    return HabitatSimActions.STOP


def log_action_data(data, i):
    if data.action != "stepPhysics":
        print("Action: {} - {}".format(data.action, i))
    else:
        print("Action {} - {}".format(data.action, i))


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


def run_reference_replay(cfg, restore_state=False, step_env=False, log_action=False, num_episodes=None, output_prefix=None):
    instructions = []
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        obs_list = []
        total_success = 0
        total_spl = 0
        total_coverage = 0
        visible_area = 0

        num_episodes = len(env.episodes)

        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            observation_list = []
            top_down_list = []

            obs = env.reset()
            if ep_id <3:
                continue

            print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            physics_simulation_library = env._sim.get_physics_simulation_library()
            print("Physics simulation library: {}".format(physics_simulation_library))
            print("Episode length: {}, Episode index: {}".format(len(env.current_episode.reference_replay), ep_id))
            print("Scene Id : {}".format(env.current_episode.scene_id))
            i = 0
            data = {
                "episodeId": env.current_episode.episode_id,
                "sceneId": env.current_episode.scene_id,
                "video": "{}_{}.mp4".format(output_prefix, ep_id),
                "task": env.current_episode.instruction.instruction_text,
                "episodeLength": len(env.current_episode.reference_replay)
            }
            instructions.append(data)
            step_index = 1
            grab_seen = False
            grab_count = 0
            success = 0
            total_reward = 0.0
            episode = env.current_episode

            for data in env.current_episode.reference_replay[step_index:]:
                
                if log_action:
                    log_action_data(data, i)
                action = possible_actions.index(data.action)
                # action = possible_actions.index(episode.actions[i])
                # action = get_habitat_sim_action(data)
                action_name = env.task.get_action_name(
                    action
                )
                #print(action_name)

                if step_env:
                    observations = env.step(action=action)
                elif not restore_state:
                    observations = env.step(action=action, replay_data=data)
                else:
                    agent_state = data.agent_state
                    sensor_states = data.agent_state.sensor_data
                    object_states = data.object_states
                    observations = env._sim.get_observations_at(agent_state.position, agent_state.rotation, sensor_states, object_states)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, {})
                top_down_frame = observations_to_image({"rgb": observations["rgb"]}, info, top_down_map_only=True)
                #frame = append_text_to_image(frame, "Instruction: {}".format(env.current_episode.instruction.instruction_text))
                total_reward += info["rearrangement_reward"]
                success = info["success"]

                observation_list.append(frame)
                top_down_list.append(top_down_frame)
                i+=1
            
            if len(episode.reference_replay) < 2000:
                total_success += success
                total_spl += info["spl"]
            # visible_area += get_visible_area(info["top_down_map"])
            # total_coverage += get_coverage(info["top_down_map"])
            # save_image(frame, "s_path_{}.png".format(ep_id))
            make_videos([observation_list], output_prefix, ep_id)
            make_videos([top_down_list], "{}_top_down".format(output_prefix), ep_id)
            print("Total reward for trajectory: {} - {}".format(total_reward, success))
            # break

        print("split: {}".format(cfg.DATASET.DATA_PATH))
        print("Average success: {} - {} - {}".format(total_success / num_episodes, total_success, num_episodes))
        print("Average SPL: {} - {} - {}".format(total_spl / num_episodes, total_spl, num_episodes))
        print("Average Coverage: {}, {}, {}".format(total_coverage/num_episodes, total_coverage, num_episodes))
        print("Average visible area: {}, {}, {}".format(visible_area/num_episodes, visible_area, num_episodes))

        if os.path.isfile("instructions.json"):
            inst_file = open("instructions.json", "r")
            existing_instructions = json.loads(inst_file.read())
            instructions.extend(existing_instructions)

        # inst_file = open("instructions.json", "w")
        # inst_file.write(json.dumps(instructions))
        return obs_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-episode", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="demo"
    )
    parser.add_argument(
        "--restore-state", dest='restore_state', action='store_true'
    )
    parser.add_argument(
        "--step-env", dest='step_env', action='store_true'
    )
    parser.add_argument(
        "--log-action", dest='log_action', action='store_true'
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.replay_episode
    # cfg.DATASET.CONTENT_SCENES = ["S9hNv5qa7GM"]
    cfg.freeze()

    observations = run_reference_replay(
        cfg, args.restore_state, args.step_env, args.log_action,
        num_episodes=1, output_prefix=args.output_prefix
    )

if __name__ == "__main__":
    main()
