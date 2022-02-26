import msgpack_numpy
import os
import torch
from collections import defaultdict
from typing import List

import lmdb
import magnum as mn
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat.tasks.pickplace.pickplace import RearrangementEpisode


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
    
    observations_batch = list(transposed[1])
    next_actions_batch = list(transposed[2])
    prev_actions_batch = list(transposed[3])
    weights_batch = list(transposed[4])
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
        next_actions_batch[bid] = _pad_helper(next_actions_batch[bid], max_traj_len)
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
    
    next_actions_batch = torch.stack(next_actions_batch, dim=1)
    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(next_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch,
        not_done_masks,
        next_actions_batch,
        weights_batch,
    )


class PickPlaceDataset(Dataset):
    """Pytorch dataset for object rearrangement task for each episode"""

    def __init__(self, config, content_scenes=["*"], mode="train", use_iw=False, inflection_weight_coef=1.0):
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
        self.dataset_path = config.DATASET_PATH.format(split=mode, scene_split=scene_split_name)

        self.config.defrost()
        self.config.DATASET.CONTENT_SCENES = content_scenes
        self.config.freeze()

        self.resolution = [self.config.SIMULATOR.RGB_SENSOR.WIDTH, self.config.SIMULATOR.RGB_SENSOR.HEIGHT]
        self.possible_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        self.total_actions = 0
        self.inflections = 0
        self.inflection_weight_coef = inflection_weight_coef

        if use_iw:
            self.inflec_weight = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weight = torch.tensor([1.0, 1.0])

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """
            self.env = habitat.Env(config=self.config)
            self.episodes = self.env._dataset.episodes
            self.instruction_vocab = self.env._dataset.instruction_vocab

            logger.info(
                "Dataset cache not found. Saving rgb, seg, depth scene images"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )

            self.scene_ids = []
            self.scene_episode_dict = {}

            # dict for storing list of episodes for each scene
            for episode in self.episodes:
                if episode.scene_id not in self.scene_ids:
                    self.scene_ids.append(episode.scene_id)
                    self.scene_episode_dict[episode.scene_id] = [episode]
                else:
                    self.scene_episode_dict[episode.scene_id].append(episode)

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(2e12),
                writemap=True,
            )

            self.count = 0
            for scene in tqdm(list(self.scene_episode_dict.keys())):
                for episode in tqdm(self.scene_episode_dict[scene]):
                    self.load_scene(scene, episode)
                    state_index_queue = []
                    try:
                        # TODO: Consider alternative for shortest_paths
                        state_index_queue.extend(range(0, len(episode.reference_replay) - 1))
                    except AttributeError as e:
                        logger.error(e)
                    self.save_frames(state_index_queue, episode)
            print("Inflection weight coef: {}, N: {}, nI: {}".format(self.total_actions / self.inflections, self.total_actions, self.inflections))
            logger.info("Rearrangement database ready!")
            self.env.close()
        else:
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 4)
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
        observations = {
            "rgb": [],
            "depth": [],
            "instruction": [],
        }
        reference_replay = episode.reference_replay
        instruction = episode.instruction
        print("Replay len: {}".format(len(reference_replay)))
        for state_index in state_index_queue:
            instruction_tokens = np.array(instruction.instruction_tokens)

            state = reference_replay[state_index]
            position = state.agent_state.position
            rotation = state.agent_state.rotation
            object_states = state.object_states
            sensor_states = state.agent_state.sensor_data

            observation = self.env.sim.get_observations_at(
                position, rotation, sensor_states, object_states
            )

            next_state = reference_replay[state_index + 1]
            next_action = self.possible_actions.index(next_state.action)

            prev_state = reference_replay[state_index]
            prev_action = self.possible_actions.index(prev_state.action)

            observations["depth"].append(observation["depth"])
            observations["rgb"].append(observation["rgb"])
            observations["instruction"].append(instruction_tokens)
            next_actions.append(next_action)
            prev_actions.append(prev_action)
        
        oracle_actions = np.array(next_actions)
        inflection_weights = np.concatenate(([1], oracle_actions[1:] != oracle_actions[:-1]))
        self.total_actions += inflection_weights.shape[0]
        self.inflections += np.sum(inflection_weights)
        inflection_weights = self.inflec_weight[torch.from_numpy(inflection_weights)].numpy()

        sample_key = "{0:0=6d}".format(self.count)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
            txn.put((sample_key + "_next_action").encode(), np.array(next_actions).tobytes())
            txn.put((sample_key + "_prev_action").encode(), np.array(prev_actions).tobytes())
            txn.put((sample_key + "_weights").encode(), inflection_weights.tobytes())
        
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

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(2e12),
                writemap=True,
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

        weight_idx = "{0:0=6d}_weights".format(idx)
        weight_binary = self.lmdb_cursor.get(weight_idx.encode())
        weight = np.frombuffer(weight_binary, dtype="float32")
        weight = torch.from_numpy(np.copy(weight))
        weight = torch.where(weight != 1.0, self.inflection_weight_coef, 1.0)

        return idx, observations, next_action, prev_action, weight
