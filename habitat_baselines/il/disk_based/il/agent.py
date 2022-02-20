#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy


class BCAgent(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_mini_batch: int,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.model = model

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(model.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        total_loss_epoch = 0.0
        action_loss_epoch = 0.0

        profiling_wrapper.range_push("PPO.update epoch")
        data_generator = rollouts.recurrent_generator(
            self.num_mini_batch
        )

        for sample in data_generator:
            (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                masks_batch,
                old_action_log_probs_batch,
            ) = sample

            # Reshape to do in a single forward pass for all steps
            (
                logits,
                rnn_hidden_states,
            ) = self.model(
                obs_batch,
                recurrent_hidden_states_batch,
                prev_actions_batch,
                masks_batch,
            )

            action_loss = cross_entropy_loss(logits, actions_batch).mean()

            self.optimizer.zero_grad()
            inflections_batch = obs_batch["inflection_weight"]
            total_loss = (action_loss * inflections_batch) / inflections_batch.sum(0)

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            action_loss_epoch += action_loss.item()
            total_loss_epoch += total_loss.item()

        profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.num_mini_batch

        action_loss_epoch /= num_updates
        total_loss_epoch /= num_updates

        return action_loss_epoch, total_loss_epochs, rnn_hidden_states

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
