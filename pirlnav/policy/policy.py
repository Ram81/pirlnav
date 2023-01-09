#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

from habitat import logger
from habitat_baselines.rl.ppo import Policy
from habitat_baselines.utils.common import CategoricalNet
from torch import nn as nn


class ILPolicy(nn.Module, Policy):
    def __init__(
        self,
        net,
        dim_actions,
        no_critic=False,
        mlp_critic=False,
        critic_hidden_dim=512,
    ):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.no_critic = no_critic

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        if self.no_critic:
            self.critic = None
        else:
            if not mlp_critic:
                self.critic = CriticHead(self.net.output_size)
            else:
                self.critic = MLPCriticHead(
                    self.net.output_size,
                    critic_hidden_dim,
                )

    def forward(self, *x):
        features, rnn_hidden_states = self.net(*x)
        distribution = self.action_distribution(features)
        distribution_entropy = distribution.entropy().mean()

        return distribution.logits, rnn_hidden_states, distribution_entropy

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        return_distribution=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        if self.no_critic:
            return action, rnn_hidden_states

        value = self.critic(features)

        if return_distribution:
            return (
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
            )

        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
        )

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class MLPCriticHead(nn.Module):
    def __init__(self, input_size, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.orthogonal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, 0)

        nn.init.orthogonal_(self.fc[2].weight)
        nn.init.constant_(self.fc[2].bias, 0)

    def forward(self, x):
        return self.fc(x.detach())
