#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces, Space
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy
from habitat_baselines.utils.common import Flatten
from habitat_baselines.il.disk_based.models.seq_2_seq_model import Seq2SeqNet


@baseline_registry.register_policy
class RearrangementSeqNetPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )
