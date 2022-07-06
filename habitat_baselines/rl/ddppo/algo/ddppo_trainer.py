#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
import wandb
from collections import defaultdict, deque
from typing import DefaultDict, Optional

import numpy as np
import torch
from torch import distributed as distrib
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
    SLURM_JOBID,
)
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import batch_obs, linear_decay, linear_warmup, critic_linear_decay
from habitat_baselines.utils.env_utils import construct_envs
# from habitat_baselines.il.env_based.policy.rednet import load_rednet
from habitat_baselines.il.env_based.policy.semantic_predictor import SemanticPredictor


@baseline_registry.register_trainer(name="ddppo")
class DDPPOTrainer(PPOTrainer):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config: Optional[Config] = None) -> None:
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = self.envs.observation_spaces[0]
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        self.semantic_predictor = None
        model_config = self.config.MODEL
        self.use_pred_semantic = hasattr(self.config.MODEL, "USE_PRED_SEMANTICS") and self.config.MODEL.USE_PRED_SEMANTICS
        # if self.use_pred_semantic:
        #     self.semantic_predictor = load_rednet(
        #         self.device,
        #         ckpt=self.config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
        #         resize=True, # since we train on half-vision
        #         num_classes=self.config.MODEL.SEMANTIC_ENCODER.num_classes
        #     )
        #     self.semantic_predictor.eval()
        if self.use_pred_semantic:
            # self.semantic_predictor = load_rednet(
            #     self.device,
            #     ckpt=model_config.SEMANTIC_ENCODER.rednet_ckpt,
            #     resize=True, # since we train on half-vision
            #     num_classes=model_config.SEMANTIC_ENCODER.num_classes
            # )
            self.semantic_predictor = SemanticPredictor(model_config, self.device)
            self.semantic_predictor.eval()

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
            logger.info("Loading state")
        
        if self.config.RL.DDPPO.pretrained:
            missing_keys = self.actor_critic.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in pretrained_state["state_dict"].items()
                }, strict=False
            )
            logger.info("Loading checkpoint missing keys: {}".format(missing_keys))

        self.rl_finetuning = False
        if hasattr(self.config.RL, "Finetune") and self.config.RL.Finetune.finetune:
            logger.info("Start Freeze encoder")
            self.rl_finetuning = True
            if self.config.RL.Finetune.freeze_encoders:
                self.actor_critic.freeze_visual_encoders()

            self.actor_finetuning_update = self.config.RL.Finetune.start_actor_finetuning_at
            self.actor_lr_warmup_update = self.config.RL.Finetune.actor_lr_warmup_update
            self.critic_lr_decay_update = self.config.RL.Finetune.critic_lr_decay_update
            self.start_critic_warmup_at = self.config.RL.Finetune.start_critic_warmup_at

        if self.config.RL.DDPPO.reset_critic:
            pass
            # nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            # nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            finetune=self.rl_finetuning,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )
        interrupted_state_file = os.path.join(self.config.CHECKPOINT_FOLDER, "{}.pth".format(SLURM_JOBID))

        interrupted_state = load_interrupted_state(interrupted_state_file)
        if interrupted_state is not None:
            logger.info("Overriding current config with interrupted state config")
            self.config = interrupted_state["config"]

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()
        self.current_update = 0

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            workers_ignore_signals=True,
        )

        ppo_cfg = self.config.RL.PPO
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )
            # self.init_wandb(interrupted_state is not None)


        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        if self.use_pred_semantic:
            batch["semantic"] = self.semantic_predictor(batch)

        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        obs_space = self.obs_space

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_values = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_gae = dict(
            steps=torch.zeros(self.envs.num_envs, 1, device=self.device),
            sum=torch.zeros(self.envs.num_envs, 1, device=self.device),
            min=torch.ones(self.envs.num_envs, 1, device=self.device) * 100.0,
            max=torch.ones(self.envs.num_envs, 1, device=self.device) * -100.0,
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
            values=torch.zeros(self.envs.num_envs, 1, device=self.device),
            values_gae=torch.zeros(self.envs.num_envs, 1, device=self.device), 
            values_mean=torch.zeros(self.envs.num_envs, 1, device=self.device),
            values_min=torch.zeros(self.envs.num_envs, 1, device=self.device), 
            values_max=torch.zeros(self.envs.num_envs, 1, device=self.device), 
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        if self.rl_finetuning:
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=[
                    lambda x: critic_linear_decay(x, self.start_critic_warmup_at, self.critic_lr_decay_update, self.config.RL.PPO.lr, self.config.RL.Finetune.policy_ft_lr),
                    lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
                    lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
                ]
            )
        else:
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
            )

        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]
            self.current_update = start_update

            if self.rl_finetuning and self.current_update >= self.actor_finetuning_update:
                for param in self.actor_critic.action_distribution.parameters():
                    param.requires_grad_(True)
                for param in self.actor_critic.net.state_encoder.parameters():
                    param.requires_grad_(True)      
                logger.info("unfreezing params")

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")
                self.current_update += 1

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    if self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        logger.info("save interrupted state at: {}".format(interrupted_state_file))
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            interrupted_state_file
                        )

                    requeue_job()
                    return

                # Enable actor finetuning at update actor_finetuning_update
                if self.rl_finetuning and self.current_update == self.actor_finetuning_update:
                    for param in self.actor_critic.action_distribution.parameters():
                        param.requires_grad_(True)
                    for param in self.actor_critic.net.state_encoder.parameters():
                        param.requires_grad_(True)                    
                    for i, param_group in enumerate(self.agent.optimizer.param_groups):
                        param_group["eps"] = self.config.RL.PPO.eps
                        lr_scheduler.base_lrs[i] = 1.0

                    if self.world_rank == 0:
                        logger.info("Start actor finetuning at: {}".format(self.current_update))

                        logger.info(
                            "updated agent number of parameters: {}".format(
                                sum(param.numel() if param.requires_grad else 0 for param in self.agent.parameters())
                            )
                        )
                if self.rl_finetuning and self.current_update == self.start_critic_warmup_at:
                    self.agent.optimizer.param_groups[0]["eps"] = self.config.RL.PPO.eps
                    lr_scheduler.base_lrs[0] = 1.0
                    if self.world_rank == 0:
                        logger.info("Set critic LR at: {}".format(self.current_update))


                count_steps_delta = 0
                self.agent.eval()
                profiling_wrapper.range_push("rollouts loop")
                for step in range(ppo_cfg.num_steps):

                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward,
                        running_episode_stats, current_episode_values
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break
                profiling_wrapper.range_pop()  # rollouts loop

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                    grad_norm,
                ) = self._update_agent(ppo_cfg, rollouts, current_episode_gae, running_episode_stats)
                pth_time += delta_pth_time

                stats_ordering = sorted(running_episode_stats.keys())
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, count_steps_delta, dist_entropy, grad_norm],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[2].long().item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                    ]
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    lrs = {}
                    for i, param_group in enumerate(self.agent.optimizer.param_groups):
                        lrs["pg_{}".format(i)] = param_group["lr"]

                    writer.add_scalar(
                        "reward",
                        deltas["reward"] / deltas["count"],
                        count_steps,
                    )
                    writer.add_scalars("learning_rate", lrs, count_steps)

                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        writer.add_scalars("metrics", metrics, count_steps)

                    losses = {k: l for l, k in zip(losses, ["value", "policy"])}
                    writer.add_scalars("losses", losses, count_steps)

                    writer.add_scalar("entropy", stats[3].item() / self.world_size, count_steps)
                    writer.add_scalar("grad_norm", stats[4].item() / self.world_size, count_steps)                    

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                count_steps
                                / ((time.time() - t_start) + prev_time),
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )
                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )
                        logger.info(
                            "update: {}\tLR: {}\tPG_LR: {}".format(
                                update,
                                lr_scheduler.get_lr(),
                                [param_group["lr"] for param_group in self.agent.optimizer.param_groups],
                            )
                        )

                    # checkpoint model
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

            if self.world_rank == 0:
                requeue_stats = dict(
                    env_time=env_time,
                    pth_time=pth_time,
                    count_steps=count_steps,
                    count_checkpoints=count_checkpoints,
                    start_update=update,
                    prev_time=(time.time() - t_start) + prev_time,
                )
                logger.info("save interrupted state at end: {}".format(interrupted_state_file))
                save_interrupted_state(
                    dict(
                        state_dict=self.agent.state_dict(),
                        optim_state=self.agent.optimizer.state_dict(),
                        lr_sched_state=lr_scheduler.state_dict(),
                        config=self.config,
                        requeue_stats=requeue_stats,
                    ),
                    interrupted_state_file
                )

            requeue_job()
            return
