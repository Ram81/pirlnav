import math
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat_baselines.il.disk_based.models.encoders.instruction import InstructionEncoder
from habitat_baselines.il.disk_based.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
)
from habitat_baselines.il.disk_based.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.utils.common import CategoricalNet, CustomFixedCategorical


class Seq2SeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
            "ResnetRGBEncoder",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
            self.rgb_encoder = SimpleRGBCNN(
                observation_space, model_config.RGB_ENCODER.output_size
            )
        elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, device
            )
        elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
            )

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = [instruction_embedding, depth_embedding, rgb_embedding]

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )
        
        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class Seq2SeqModel(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__()
        self.net = Seq2SeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
        )
        self.action_distribution = CategoricalNet(
            self.net.output_size, action_space.n
        )
        self.train()
    
    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution.logits, rnn_hidden_states