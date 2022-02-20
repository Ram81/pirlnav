#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import numpy as np
import torch


class DiscriminatorDataBuffer:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        ground_truth_buffer_path=None,
    ):
        self.ground_truth_buffer = {}
        for sensor in observation_space.spaces:
            self.ground_truth_buffer[sensor] = torch.zeros(
                size,
                num_envs,
                *observation_space.spaces[sensor].shape
            )
        
        self.experience_buffer = {}
        for sensor in observation_space.spaces:
            self.experience_buffer[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )


        self._size = size
        self._current_idx = 0
        self.step = 0
        self.num_steps = num_steps

        self.init_ground_truth_buffer()

    def to(self, device):
        for sensor in self.experience_buffer:
            self.experience_buffer[sensor] = self.experience_buffer[sensor].to(device)
        
        for sensor in self.ground_truth_buffer:
            self.ground_truth_buffer[sensor] = self.ground_truth_buffer[sensor].to(device)

    def insert(
        self,
        observations,
    ):
        for sensor in observations:
            self.experience_buffer[sensor][self.step + 1].copy_(
                observations[sensor]
            )

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.step = 0
    
    def sample(self, batch_size, env_index):
        half_batch_size = batch_size // 2

        indices = np.random.randint(self.num_steps, size=half_batch_size)
        ground_truth_sample_obs = {}
        for sensor in observation_space.spaces:
            ground_truth_sample_obs[sensor] = self.ground_truth_buffer[sensor][indices, env_index]

        indices = np.random.randint(self.num_steps, size=half_batch_size)
        experience_sample_obs = {}
        for sensor in observation_space.spaces:
            experience_sample_obs[sensor] = self.observations[sensor][indices, env_index]
        
    
    def init_ground_truth_buffer(self):
        pass

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
