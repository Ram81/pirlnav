import cv2
import msgpack_numpy
import os
import random
import sys
import torch
from collections import defaultdict
from typing import List, Dict

import lmdb
import magnum as mn
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrangement.rearrangement import InstructionData, RearrangementEpisode
from habitat.tasks.utils import get_habitat_sim_action
from habitat.utils.visualizations.utils import observations_to_image, images_to_video
from habitat_sim.utils import viz_utils as vut
from habitat_baselines.common.environments import get_env_class


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))
    
    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    next_actions_batch = list(transposed[2])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)

    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        next_actions_batch[bid] = _pad_helper(next_actions_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)

    next_actions_batch = torch.stack(next_actions_batch, dim=1)
    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    not_done_masks = torch.ones_like(next_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch,
        not_done_masks,
    )


class RearrangementGoalDataset(Dataset):
    """Pytorch dataset for object rearrangement task for each episode"""

    def __init__(self, config, mode="train_goals"):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        self.config = config.TASK_CONFIG
        self.dataset_path = config.GOAL_DATASET_PATH.format(split=mode)
        
        self.env = habitat.Env(config=self.config)
        self.episodes = self.env._dataset.episodes
        self.instruction_vocab = self.env._dataset.instruction_vocab

        self.resolution = self.env._sim.get_resolution()

        self.scene_ids = []
        self.scene_episode_dict = {}

        # dict for storing list of episodes for each scene
        for episode in self.episodes:
            if episode.scene_id not in self.scene_ids:
                self.scene_ids.append(episode.scene_id)
                self.scene_episode_dict[episode.scene_id] = [episode]
            else:
                self.scene_episode_dict[episode.scene_id].append(episode)

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """

            logger.info(
                "Dataset cache not found. Saving goal state observations"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )

            self.count = 0
            for scene in tqdm(list(self.scene_episode_dict.keys())):
                self.count = 0
                for episode in tqdm(self.scene_episode_dict[scene]):
                    self.load_scene(scene, episode)
                    state_index_queue = []
                    try:
                        # Ignore last frame as it is only used to lookup for STOP action
                        state_index_queue.extend(range(0, len(episode.reference_replay) - 1))
                    except AttributeError as e:
                        logger.error(e)
                    self.save_frames(state_index_queue, episode)
            logger.info("Rearrangement database ready!")

        else:
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )

        self.env.close()

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"])
        self.lmdb_env.close()
        self.lmdb_env = None

    def save_frames(
        self, state_index_queue: List[int], episode: RearrangementEpisode
    ) -> None:
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """
        next_actions = []
        prev_actions = []
        observations = defaultdict(list)
        obs_list = []
        reference_replay = episode.reference_replay
        instruction = episode.instruction

        state_index = state_index_queue[-1]
        scene_id = episode.scene_id
    
        instruction_tokens = np.array(instruction.instruction_tokens)

        state = reference_replay[state_index]
        position = state.agent_state.position
        rotation = state.agent_state.rotation
        object_states = state.object_states
        sensor_states = state.agent_state.sensor_data

        observation = self.env.sim.get_observations_at(
            position, rotation, sensor_states, object_states
        )

        observations["depth"].append(observation["depth"])
        observations["rgb"].append(observation["rgb"])
        observations["instruction"].append(instruction_tokens)
        observations["demonstration"].append(0)
        observations["gripped_object_id"].append(-1)

        frame = observations_to_image(
            {"rgb": observation["rgb"]}, {}
        )
        obs_list.append(frame)

        scene_id = scene_id.split("/")[-1].replace(".", "_")
        sample_key = "{0}_{1:0=6d}".format(scene_id, self.count)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
        
        self.count += 1
        # images_to_video(images=obs_list, output_dir="demos", video_name="dummy_{}".format(self.count))

    def cache_exists(self) -> bool:
        if os.path.exists(self.dataset_path):
            if os.listdir(self.dataset_path):
                return True
        else:
            os.makedirs(self.dataset_path)
        return False
    
    def get_vocab_dict(self) -> VocabDict:
        r"""Returns Instruction VocabDicts"""
        return self.instruction_vocab

    def load_scene(self, scene, episode) -> None:
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.SIMULATOR.objects = episode.objects
        self.config.freeze()
        self.env.sim.reconfigure(self.config.SIMULATOR)
    
    def get_scene_episode_length(self, scene: str) -> int:
        return len(self.scene_episode_dict[scene])

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                readonly=True,
                lock=False,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])
    
        obs_idx = "{0}{1:0=6d}_obs".format(scene_id, idx)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        return observations
    
    def get_item(self, idx: int, scene_id: str):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        org_scene_id = scene_id
        scene_id = scene_id.split("/")[-1].replace(".", "_")
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                readonly=True,
                lock=False,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])

        obs_idx = "{0}_{1:0=6d}_obs".format(scene_id, idx)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        return observations

    def get_items(self, idxs: int, scene_ids: str):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        org_scene_id = scene_id
        scene_id = scene_id.split("/")[-1].replace(".", "_")
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                readonly=True,
                lock=False,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])
        batch_obs = defaultdict(list)
        for idx, scene_id in zip(idxs, scene_ids):
            obs_idx = "{0}_{1:0=6d}_obs".format(scene_id, idx)
            observations_binary = self.lmdb_cursor.get(obs_idx.encode())
            observations = msgpack_numpy.unpackb(observations_binary, raw=False)
            for k, v in observations.items():
                obs = np.array(observations[k])
                obs = torch.from_numpy(obs)
                batch_obs[k].append(obs)
        
        for key, val in batch_obs:
            batch_obs[key] = torch.stack(batch_obs[key], 1)

        return batch_obs


class RearrangementGoalDatasetV2(Dataset):
    """Pytorch dataset for object rearrangement task for each episode"""

    def __init__(self, config, mode="train_goals", content_scenes=["*"]):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        scene_split_name = "train"
        if content_scenes[0] != "*":
            scene_split_name = "_".join(content_scenes)

        self.config = config.TASK_CONFIG
        self.dataset_path = config.GOAL_DATASET_PATH.format(split=mode, scene_split=scene_split_name)

        self.config.defrost()
        self.config.DATASET.CONTENT_SCENES = content_scenes
        self.config.freeze()

        self.env = habitat.Env(config=self.config)
        self.episodes = self.env._dataset.episodes
        self.instruction_vocab = self.env._dataset.instruction_vocab

        self.resolution = self.env._sim.get_resolution()
        self.possible_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        self.count = 0
        self.scene_episode_dict = defaultdict(list)
        self.scene_episode_idx_dict = defaultdict(int)

        # dict for storing list of episodes for each scene
        for episode in self.episodes:
            self.scene_episode_dict[episode.scene_id].append(episode)
        self.scene_ids = list(self.scene_episode_dict.keys())

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """

            logger.info(
                "Dataset cache not found/ignored. Saving goal state observations"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )

            for i, episode in enumerate(self.episodes):
                self.env.reset()
                episode = self.env.current_episode
                state_index_queue = []
                try:
                    # Ignore last frame as it is only used to lookup for STOP action
                    state_index_queue.extend(range(0, len(episode.reference_replay) - 1))
                except AttributeError as e:
                    logger.error(e)
                self.save_frames(state_index_queue, episode)
            logger.info("Rearrangement database ready!")

        else:
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )

        self.env.close()

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 3)
        self.lmdb_env.close()
        self.lmdb_env = None

    def save_frames(
        self, state_index_queue: List[int], episode: RearrangementEpisode
    ) -> None:
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """
        next_actions = []
        prev_actions = []
        observations = defaultdict(list)
        reference_replay = episode.reference_replay
        instruction = episode.instruction
        obs_list = []
    
        instruction_tokens = np.array(instruction.instruction_tokens)
        scene_id = episode.scene_id

        for state_index in state_index_queue[1:]:
            instruction_tokens = np.array(instruction.instruction_tokens)

            state = reference_replay[state_index]
            position = state.agent_state.position
            rotation = state.agent_state.rotation
            object_states = state.object_states
            sensor_states = state.agent_state.sensor_data

            action = self.possible_actions.index(state.action)

            observation = self.env.step(action=action, replay_data=state)

            next_state = reference_replay[state_index + 1]
            next_action = self.possible_actions.index(next_state.action)

            prev_state = reference_replay[state_index]
            prev_action = self.possible_actions.index(prev_state.action)

            observations["compass"].append(observation["compass"])
            observations["gps"].append(observation["gps"])
            observations["all_object_positions"].append(observation["all_object_positions"])
            observations["instruction"].append(instruction_tokens)
            next_actions.append(next_action)
            prev_actions.append(prev_action)

            frame = observations_to_image(
                {"rgb": observation["rgb"]}, {}
            )
            obs_list.append(frame)

        # count = self.scene_episode_idx_dict[scene_id]
        # self.scene_episode_idx_dict[scene_id] += 1
        # scene_id = scene_id.split("/")[-1].replace(".", "_")
        # sample_key = "{0}_{1:0=6d}".format(scene_id, count)
        logger.info("Episode: {} - {}".format(self.count, len(observations["compass"])))
        sample_key = "{0:0=6d}".format(self.count)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
            txn.put((sample_key + "_next_action").encode(), np.array(next_actions).tobytes())
            txn.put((sample_key + "_prev_action").encode(), np.array(prev_actions).tobytes())

        self.count += 1

        # images_to_video(images=obs_list, output_dir="demos", video_name="dummy_{}_{}".format(scene_id, count))

    def cache_exists(self) -> bool:
        if os.path.exists(self.dataset_path):
            if os.listdir(self.dataset_path):
                return True
        else:
            os.makedirs(self.dataset_path)
        return False
    
    def get_vocab_dict(self) -> VocabDict:
        r"""Returns Instruction VocabDicts"""
        return self.instruction_vocab

    def load_scene(self, scene, episode) -> None:
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.SIMULATOR.objects = episode.objects
        self.config.freeze()
        self.env.sim.reconfigure(self.config.SIMULATOR)
    
    def get_scene_episode_length(self, scene: str) -> int:
        return len(self.scene_episode_dict[scene])

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                readonly=True,
                lock=False,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])

        obs_idx = "{0:0=6d}_obs".format(idx)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        next_action_idx = "{0:0=6d}_next_action".format(idx)
        next_action_binary = self.lmdb_cursor.get(next_action_idx.encode())
        next_action = np.frombuffer(next_action_binary, dtype="int")
        next_action = torch.from_numpy(np.copy(next_action))

        prev_action_idx = "{0:0=6d}_prev_action".format(idx)
        prev_action_binary = self.lmdb_cursor.get(prev_action_idx.encode())
        prev_action = np.frombuffer(prev_action_binary, dtype="int")
        prev_action = torch.from_numpy(np.copy(prev_action))

        return observations, prev_action, next_action
    
    def get_item(self, idx: int, scene_id: str):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        org_scene_id = scene_id
        scene_id = scene_id.split("/")[-1].replace(".", "_")
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                readonly=True,
                lock=False,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])

        obs_idx = "{0}_{1:0=6d}_obs".format(scene_id, idx)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        next_action_idx = "{0}_{1:0=6d}_next_action".format(scene_id, idx)
        next_action_binary = self.lmdb_cursor.get(next_action_idx.encode())
        next_action = np.frombuffer(next_action_binary, dtype="int")
        next_action = torch.from_numpy(np.copy(next_action))

        prev_action_idx = "{0}_{1:0=6d}_prev_action".format(scene_id, idx)
        prev_action_binary = self.lmdb_cursor.get(prev_action_idx.encode())
        prev_action = np.frombuffer(prev_action_binary, dtype="int")
        prev_action = torch.from_numpy(np.copy(prev_action))

        return observations, prev_action, next_action
