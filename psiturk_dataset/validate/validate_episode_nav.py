import argparse
import cv2
import habitat
import json
import sys
import time
import os
import math
import numpy as np

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from psiturk_dataset.utils.utils import write_json


def are_points_navigable(sim, points):
    pathfinder = sim.pathfinder
    is_navigable_list = []
    for point in points:
        is_navigable_list.append(pathfinder.is_navigable(point))

    for i in range(len(points)):
        for j in range(len(points)):
            if i <= j:
                continue
            dist = sim.geodesic_distance(points[i], points[j])
            if dist == np.inf or dist == math.inf:
                return False
    
    if np.sum(is_navigable_list) != len(is_navigable_list):
        return False
    return True


def are_goals_reachable(sim, points, goals):
    pathfinder = sim.pathfinder
    is_navigable_list = []
    for point in points:
        is_navigable_list.append(pathfinder.is_navigable(point))
    
    # for point in goals:
    #     is_navigable_list.append(pathfinder.is_navigable(point))
    any_reachable = False
    for i in range(len(points)):
        for j in range(len(goals)):
            dist = sim.geodesic_distance(points[i], goals[j])
            if dist != np.inf or dist != math.inf:
                any_reachable = True
    
    if np.sum(is_navigable_list) != len(is_navigable_list):
        return False
    return any_reachable


def get_object_and_agent_state(sim, episode):
    points = []
    # Append agent state
    agent_position = sim.get_agent_state().position
    points.append(agent_position)

    # Append object state
    object_ids = sim.get_existing_object_ids()
    for object_id in object_ids:
        points.append(sim.get_translation(object_id))
    
    # Append goal state
    goals = []
    for goal in episode.goals:
        goals.append(goal.position)
    
    return points, goals


def run_validation(cfg, num_steps=5):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        obs_list = []
        non_navigable_episodes = []
        navigable_episodes = 0

        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            observation_list = []

            obs = env.reset()

            # print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            # physics_simulation_library = env._sim.get_physics_simulation_library()
            # print("Physics simulation library: {}".format(physics_simulation_library))
            # print("Scene Id : {}".format(env.current_episode.scene_id))
            
            # action = possible_actions.index("NO_OP")
            # for i in range(num_steps):
            #     observations = env.step(action=action)
            sim = env._sim
            points, goals = get_object_and_agent_state(sim, env.current_episode)
            is_navigable = are_points_navigable(sim, points)
            is_reachable = are_goals_reachable(sim, points, goals)
            navigable_episodes += int(is_navigable and is_reachable)

            if not is_navigable or not is_reachable:
                non_navigable_episodes.append(env.current_episode.episode_id)

            if ep_id % 10 == 0:
                print("Total {}/{} episodes are navigable".format(navigable_episodes, len(env.episodes)))
        print("Total {}/{} episodes are navigable".format(navigable_episodes, len(env.episodes)))
        print(non_navigable_episodes)
        write_json(non_navigable_episodes, "data/hit_data/non_navigable_episodes_ll.json")


def validate_objectnav_episodes(cfg):
    with habitat.Env(cfg) as env:
        easy_episodes = []
        counter = 0
        print("Total episodes: {}".format(len(env.episodes)))
        visited_eps = {}
        duplicates = 0
        for ep_id in range(len(env.episodes)):
            obs = env.reset()

            info = env.get_metrics()
            current_episode = env._current_episode
            episode_key = str(current_episode.start_position) + "_{}".format(current_episode.scene_id)
            if visited_eps.get(episode_key) != 1:
                visited_eps[episode_key] = 1
            else:
                duplicates += 1

            if info["distance_to_goal"] >= 15.0:
                easy_episodes.append(env.current_episode.episode_id)
                # print(info["distance_to_goal"])
            counter += 1

            if ep_id % 10 == 0:
                print("Total {}/{} episodes are navigable".format(len(easy_episodes), counter))
                print("Total {}/{} episodes are duplicates".format(duplicates, counter))
        print("Total {}/{} episodes are navigable".format(len(easy_episodes), len(env.episodes)))
        print(easy_episodes)
        write_json(easy_episodes, "data/hit_data/easy_episodes_thda.json")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--num-steps", type=int, default=5
    )
    parser.add_argument(
        "--config", type=str, default="configs/tasks/object_rearrangement.yaml"
    )
    parser.add_argument(
        "--objectnav", dest="is_objectnav", action="store_true"
    )
    args = parser.parse_args()
    config = habitat.get_config(args.config)
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.episodes
    cfg.TASK.SENSORS = ['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']
    cfg.DATASET.CONTENT_SCENES = ["XcA2TqTSSAj"]
    cfg.freeze()
    print(args)
    if args.is_objectnav:
        validate_objectnav_episodes(cfg)
    else:
        run_validation(cfg, args.num_steps)

if __name__ == "__main__":
    main()
