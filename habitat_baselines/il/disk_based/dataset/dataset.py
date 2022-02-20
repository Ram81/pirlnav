import cv2
import os
import random
import sys
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
from habitat.tasks.rearrangement.rearrangement import InstructionData
from habitat.tasks.utils import get_habitat_sim_action
from habitat.utils.visualizations.utils import observations_to_image, images_to_video
from habitat_sim.utils import viz_utils as vut


class RearrangementDataset(Dataset):
    """Pytorch dataset for object rearrangement task"""

    def __init__(self, config, mode="train"):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        self.config = config.TASK_CONFIG
        self.dataset_path = config.DATASET_PATH.format(split=mode)
        
        self.env = habitat.Env(config=self.config)
        self.episodes = self.env._dataset.episodes
        self.instruction_vocab = self.env._dataset.instruction_vocab

        self.resolution = self.env._sim.get_resolution()

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """


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
                map_size=int(1e11),
                writemap=True,
            )

            self.count = 0

            for scene in tqdm(list(self.scene_episode_dict.keys())):
                for episode in tqdm(self.scene_episode_dict[scene]):
                    self.load_scene(scene, episode)
                    try:
                        # TODO: Consider alternative for shortest_paths
                        state_index_queue = range(0, len(episode.reference_replay))
                    except AttributeError as e:
                        logger.error(e)
                    # Sample states if needed
                    # random_states = random.sample(state_queue, 9)
                    self.save_frames(state_index_queue, episode.reference_replay, episode.instruction)

            logger.info("Rearrangement database ready!")

        else:
            print("HabitatSimActions.TURN_RIGHT", HabitatSimActions.TURN_RIGHT)
            print("HabitatSimActions.TURN_LEFT", HabitatSimActions.TURN_LEFT)
            print("HabitatSimActions.MOVE_FORWARD", HabitatSimActions.MOVE_FORWARD)
            print("HabitatSimActions.MOVE_BACKWARD", HabitatSimActions.MOVE_BACKWARD)
            print("HabitatSimActions.LOOK_UP", HabitatSimActions.LOOK_UP)
            print("HabitatSimActions.LOOK_DOWN", HabitatSimActions.LOOK_DOWN)
            print("HabitatSimActions.NO_OP", HabitatSimActions.NO_OP)
            print("HabitatSimActions.GRAB_RELEASE", HabitatSimActions.GRAB_RELEASE)
            print("HabitatSimActions.START", HabitatSimActions.START)
            print("HabitatSimActions.STOP", HabitatSimActions.STOP)
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )
        
        self.env.close()

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 7)
        self.lmdb_env.close()
        self.lmdb_env = None

    def save_frames(
        self, state_index_queue: List[int], reference_replay: List[Dict],
        instruction: InstructionData
    ) -> None:
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """
        obs_list = []
        for state_index in state_index_queue:
            state = reference_replay[state_index]
            position = state["agent_state"]["position"]
            rotation = state["agent_state"]["rotation"]
            object_states = state["object_states"]
            sensor_states = state["agent_state"]["sensor_data"]

            next_action = HabitatSimActions.STOP
            if state_index < len(reference_replay) - 1:
                next_state = reference_replay[state_index + 1]
                next_action = get_habitat_sim_action(next_state["action"])

            prev_action = HabitatSimActions.START
            if state_index != 0:
                prev_state = reference_replay[state_index - 1]
                prev_action = get_habitat_sim_action(prev_state["action"])
            
            done = 1
            if state_index == len(reference_replay) -1:
                done = 0
            
            instruction_tokens = np.array(instruction.instruction_tokens)

            observation = self.env.sim.get_observations_at(
                position, rotation, sensor_states, object_states
            )

            depth = observation["depth"]
            rgb = observation["rgb"]

            scene = self.env.sim.semantic_annotations()
            instance_id_to_label_id = {
                int(obj.id.split("_")[-1]): obj.category.index()
                for obj in scene.objects
            }
            self.mapping = np.array(
                [
                    instance_id_to_label_id[i]
                    for i in range(len(instance_id_to_label_id))
                ]
            )
            seg = np.take(self.mapping, observation["semantic"])
            seg[seg == -1] = 0
            seg = seg.astype("uint8")

            sample_key = "{0:0=6d}".format(self.count)
            with self.lmdb_env.begin(write=True) as txn:
                txn.put((sample_key + "_rgb").encode(), rgb.tobytes())
                txn.put((sample_key + "_depth").encode(), depth.tobytes())
                txn.put((sample_key + "_seg").encode(), seg.tobytes())
                txn.put((sample_key + "_instruction").encode(), instruction_tokens.tobytes())
                txn.put((sample_key + "_next_action").encode(), bytes([next_action]))
                txn.put((sample_key + "_prev_action").encode(), bytes([prev_action]))
                txn.put((sample_key + "_done").encode(), bytes([done]))

            self.count += 1
            frame = observations_to_image(
                observation, {}
            )
            obs_list.append(frame)
        images_to_video(images=obs_list, output_dir="demos", video_name="dummy")

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
                map_size=int(1e11),
                writemap=True,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])

        rgb_idx = "{0:0=6d}_rgb".format(idx)
        rgb_binary = self.lmdb_cursor.get(rgb_idx.encode())
        rgb_np = np.frombuffer(rgb_binary, dtype="uint8")
        rgb = rgb_np.reshape(height, width, 3) / 255.0
        rgb = rgb.astype(np.float32)

        depth_idx = "{0:0=6d}_depth".format(idx)
        depth_binary = self.lmdb_cursor.get(depth_idx.encode())
        depth_np = np.frombuffer(depth_binary, dtype="float32")
        depth = depth_np.reshape(height, width, 1)

        seg_idx = "{0:0=6d}_seg".format(idx)
        seg_binary = self.lmdb_cursor.get(seg_idx.encode())
        seg_np = np.frombuffer(seg_binary, dtype="uint8")
        seg = seg_np.reshape(height, width)

        instruction_idx = "{0:0=6d}_instruction".format(idx)
        instruction_binary = self.lmdb_cursor.get(instruction_idx.encode())
        instruction_tokens = np.frombuffer(instruction_binary, dtype="int")

        next_action_idx = "{0:0=6d}_next_action".format(idx)
        next_action_binary = self.lmdb_cursor.get(next_action_idx.encode())
        next_action = np.frombuffer(next_action_binary, dtype="uint8")

        prev_action_idx = "{0:0=6d}_prev_action".format(idx)
        prev_action_binary = self.lmdb_cursor.get(prev_action_idx.encode())
        prev_action = np.frombuffer(prev_action_binary, dtype="uint8")

        done_idx = "{0:0=6d}_done".format(idx)
        done_binary = self.lmdb_cursor.get(done_idx.encode())
        done = np.frombuffer(done_binary, dtype="uint8")

        return idx, rgb, depth, seg, instruction_tokens, next_action[0], prev_action[0], done[0]
