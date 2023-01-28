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
from habitat_baselines.utils.common import linear_decay

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
        finetune: bool = False,
        finetune_full_agent: bool = False,
        vpt_finetuning: bool = False,
        pretrained_policy: Policy = None,
        kl_coef: float = 0.0,
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # VPT style finetuning parameters
        self.pretrained_policy = pretrained_policy
        self.kl_coef = kl_coef
        self.vpt_finetuning = vpt_finetuning

        if not finetune:
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
                lr=lr,
                eps=eps,
            )
        elif finetune_full_agent:
            self.optimizer = optim.Adam([
                {
                    'params': list(filter(lambda p: p.requires_grad, actor_critic.critic.parameters())),
                    'lr': lr,
                    'eps': eps
                },
                {
                    'params': list(actor_critic.net.state_encoder.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.action_distribution.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.net.visual_encoder.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.net.prev_action_embedding.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.net.compass_embedding.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.net.obj_categories_embedding.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.net.gps_embedding.parameters()),
                    'lr': 0.0,
                },
            ])
        else:
            self.optimizer = optim.Adam([
                {
                    'params': list(filter(lambda p: p.requires_grad, actor_critic.critic.parameters())),
                    'lr': lr,
                    'eps': eps
                },
                {
                    'params': list(actor_critic.net.state_encoder.parameters()),
                    'lr': 0.0,
                },
                {
                    'params': list(actor_critic.action_distribution.parameters()),
                    'lr': 0.0,
                },
            ])

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        avg_grad_norm = 0.0
        aux_kl_constraint_epoch = 0.0



        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_loss_meta
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                aux_kl_constraint = 0.0
                if self.vpt_finetuning:
                    with torch.no_grad():
                        (
                            _,
                            _,
                            _,
                            _,
                            pretrained_policy_aux_loss_meta
                        ) = self.pretrained_policy.evaluate_actions(
                            obs_batch,
                            recurrent_hidden_states_batch,
                            prev_actions_batch,
                            masks_batch,
                            actions_batch,
                        )

                        aux_kl_constraint = torch.distributions.kl.kl_divergence(
                            pretrained_policy_aux_loss_meta["action_distribution"],
                            aux_loss_meta["action_distribution"]
                        ).mean()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + aux_kl_constraint * self.kl_coef
                )            

                self.before_backward(total_loss)
                total_loss.backward()
                avg_grad_norm += self.get_grad_norm()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                aux_kl_constraint_epoch += aux_kl_constraint

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        avg_grad_norm /= num_updates
        aux_kl_constraint_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, avg_grad_norm, aux_kl_constraint_epoch

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
    
    def get_grad_norm(self):
        parameters= [p for p in self.actor_critic.parameters() if p.grad is not None and p.requires_grad]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(self.device) for p in parameters]), 2)
        return total_norm