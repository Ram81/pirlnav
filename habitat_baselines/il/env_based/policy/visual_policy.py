from typing import Dict

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from gym.spaces import Dict, Box
from habitat import Config, logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoalSensor,
    ObjectGoalPromptSensor
)
from habitat_baselines.il.common.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
)
from habitat_baselines.il.common.encoders.visual_encoder import VisualEncoder
from habitat_baselines.il.common.utils import load_encoder
from habitat_baselines.il.common.transforms import get_transform
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Net, Policy


class ObjectNavMAEILNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        model_config: Config,
        num_actions,
        run_type: str,
    ):
        super().__init__()
        self.model_config = model_config
        rnn_input_size = 0
        logger.info("\n\nSetting up ObjectNavMAEILPolicy")

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder",
            "None",
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )
            rnn_input_size += model_config.DEPTH_ENCODER.output_size
        else:
            self.depth_encoder = None

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "ResnetRGBEncoder",
            "VisualEncoder",
            "None",
        ], "RGB_ENCODER.cnn_type must be 'ResnetRGBEncoder'."

        logger.info("RGB encoder is {}".format(model_config.RGB_ENCODER.cnn_type))
        rgb_config = model_config.RGB_ENCODER
        if model_config.RGB_ENCODER.cnn_type == "VisualEncoder":
            name = "resize"
            if rgb_config.use_augmentations and run_type == "train":
                name = rgb_config.augmentations_name
            if rgb_config.use_augmentations_test_time and run_type == "eval":
                name = rgb_config.augmentations_name
            self.visual_transform = get_transform(name, size=rgb_config.image_size)
            self.visual_transform.randomize_environments = rgb_config.randomize_augmentations_over_envs

            self.visual_encoder = VisualEncoder(
                image_size=rgb_config.image_size,
                backbone=rgb_config.backbone,
                input_channels=3,
                resnet_baseplanes=rgb_config.resnet_baseplanes,
                resnet_ngroups=rgb_config.resnet_baseplanes // 2,
                vit_use_fc_norm=rgb_config.vit_use_fc_norm,
                vit_global_pool=rgb_config.vit_global_pool,
                vit_use_cls=rgb_config.vit_use_cls,
                vit_mask_ratio=rgb_config.vit_mask_ratio,
                avgpooled_image=rgb_config.avgpooled_image,
                drop_path_rate=rgb_config.drop_path_rate,
            )

            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.visual_encoder.output_size, rgb_config.hidden_size),
                nn.ReLU(True),
            )
            rnn_input_size += rgb_config.hidden_size
        elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.visual_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
                normalize_visual_inputs=model_config.RGB_ENCODER.normalize_visual_inputs,
            )
            rnn_input_size += model_config.RGB_ENCODER.output_size
        else:
            self.visual_encoder = None


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
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        self.use_clip_goal = model_config.GOAL.use_clip_goal
        if not self.use_clip_goal and ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            logger.info("Object categories: {}".format(self._n_object_categories))
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")
        else:
            self.clip, _ = clip.load(model_config.GOAL.clip_model)
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

            rnn_input_size += 512
            logger.info("\n\nSetting up CLIP Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        # pretrained weights
        logger.info("encoder: {}".format(rgb_config.pretrained_encoder is not None))
        if rgb_config.pretrained_encoder is not None:
            msg = load_encoder(self.visual_encoder, rgb_config.pretrained_encoder)
            logger.info("Using weights from {}: {}".format(rgb_config.pretrained_encoder, msg))

        # freeze backbone
        if rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        self.rnn_input_size = rnn_input_size

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
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers


    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        N = rnn_hidden_states.size(1)

        x = []

        if self.depth_encoder is not None:
            depth_obs = observations["depth"]
            if len(depth_obs.size()) == 5:
                observations["depth"] = depth_obs.contiguous().view(
                    -1, depth_obs.size(2), depth_obs.size(3), depth_obs.size(4)
                )

            depth_embedding = self.depth_encoder(observations)
            x.append(depth_embedding)

        if self.visual_encoder is not None:
            rgb_obs = observations["rgb"]
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )

            # visual encoder
            rgb = observations["rgb"]
            if self.model_config.RGB_ENCODER.cnn_type == "VisualEncoder":    
                rgb = self.visual_transform(rgb, N)
                rgb = self.visual_encoder(rgb)
                rgb = self.visual_fc(rgb)
            else:
                rgb = self.visual_encoder(observations)
            x.append(rgb)


        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if not self.use_clip_goal and ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))
        else:
            goal = observations[ObjectGoalPromptSensor.cls_uuid]  # T x N x 1 x F
            # logger.info("Goal shape: {}".format(goal.shape))
            if len(goal.shape) == 4:
                goal = goal.flatten(0, 1) # TN x 1 x F
            goal = goal.flatten(0, 1)  # TN x F

            with torch.no_grad():
                goal = self.clip.encode_text(goal.long()).float()
                goal /= goal.norm(dim=-1, keepdim=True)

            x.append(goal)

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)
        
        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavMAEILPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config, run_type: str
    ):
        super().__init__(
            ObjectNavMAEILNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
                run_type=run_type,
            ),
            action_space.n,
            no_critic=model_config.CRITIC.no_critic,
            mlp_critic=model_config.CRITIC.mlp_critic,
            critic_hidden_dim=model_config.CRITIC.hidden_dim,
            detach_critic_input=model_config.CRITIC.detach_critic_input,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
            run_type="train",
        )

    def freeze_visual_encoders(self):
        if hasattr(self.net, "visual_encoder"):
            for param in self.net.visual_encoder.parameters():
                param.requires_grad_(False)
    
    def unfreeze_visual_encoders(self):
        if hasattr(self.net, "visual_encoder"):
            for param in self.net.visual_encoder.parameters():
                param.requires_grad_(True)