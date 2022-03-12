#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import sys
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabFromText
from habitat.tasks.pickplace.pickplace import (
    InstructionData,
    RearrangementEpisode,
    RearrangementSpec,
    RearrangementObjectSpec,
    ReplayActionSpec,
    GrabReleaseActionSpec,
    AgentStateSpec,
    ObjectStateSpec
)

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PickPlaceDataset-v0")
class PickPlaceDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[RearrangementEpisode]
    instruction_vocab: VocabFromText

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        max_semantic_object_id = 1 << 16
        self.instruction_vocab = VocabFromText(
           sentences=deserialized["instruction_vocab"]["sentences"]
        )

        for i, episode in enumerate(deserialized["episodes"]):
            episode["reference_replay"] = []
            episode = RearrangementEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            for i, goal in enumerate(episode.goals):
                if goal["info"]["is_receptacle"]:
                    episode.goals[1] = RearrangementSpec(**goal)
                else:
                    episode.goals[0] = RearrangementSpec(**goal)

            for i, obj in enumerate(episode.objects):
                episode.objects[i]["semantic_object_id"] = (episode.objects[i]["object_id"] + (1<<16))
                episode.objects[i] = RearrangementObjectSpec(**obj)
            if len(episode.reference_replay) > 1500:
                continue
            self.episodes.append(episode)


@registry.register_dataset(name="PickPlaceDataset-v1")
class PickPlaceDatasetV2(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[RearrangementEpisode]
    instruction_vocab: VocabFromText

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        max_semantic_object_id = 1 << 16
        self.instruction_vocab = VocabFromText(
           sentences=deserialized["instruction_vocab"]["sentences"]
        )

        for i, episode in enumerate(deserialized["episodes"]):
            episode = RearrangementEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            for i, replay_step in enumerate(episode.reference_replay):
                if "action_data" in replay_step.keys():
                    replay_step["action_data"] = GrabReleaseActionSpec(**replay_step["action_data"])
                if "agent_state" in replay_step.keys():
                    replay_step["agent_state"] = AgentStateSpec(**replay_step["agent_state"])
                if "object_states" in replay_step.keys():
                    for j in range(len(replay_step["object_states"])):
                        replay_step["object_states"][j] = ObjectStateSpec(**replay_step["object_states"][j])
                episode.reference_replay[i] = ReplayActionSpec(**replay_step)
            for i, goal in enumerate(episode.goals):
                if goal["info"]["is_receptacle"]:
                    episode.goals[1] = RearrangementSpec(**goal)
                else:
                    episode.goals[0] = RearrangementSpec(**goal)

            for i, obj in enumerate(episode.objects):
                episode.objects[i]["semantic_object_id"] = (episode.objects[i]["object_id"] + (1<<16))
                episode.objects[i] = RearrangementObjectSpec(**obj)
            
            self.episodes.append(episode)
