#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import wandb
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import torch
from torch import nn as nn
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image, append_text_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.il.env_based.policy.rednet import load_rednet
from scripts.utils.utils import write_json


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space
        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        self.actor_critic.to(self.device)

        self.semantic_predictor = None
        self.use_pred_semantic = hasattr(self.config.MODEL, "USE_PRED_SEMANTICS") and self.config.MODEL.USE_PRED_SEMANTICS
        if self.use_pred_semantic:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=self.config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True, # since we train on half-vision
                num_classes=self.config.MODEL.SEMANTIC_ENCODER.num_classes
            )
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

        logger.info("Freeze encoder")
        if hasattr(self.config.RL, "Finetune"):
            logger.info("Start Freeze encoder")
            self.warm_up_critic = True
            if self.config.RL.Finetune.freeze_encoders:
                self.actor_critic.freeze_visual_encoders()

            self.actor_finetuning_update = self.config.RL.Finetune.start_actor_finetuning_at
            self.actor_lr_warmup_update = self.config.RL.Finetune.actor_lr_warmup_update
            self.critic_lr_decay_update = self.config.RL.Finetune.critic_lr_decay_update
            self.start_critic_warmup_at = self.config.RL.Finetune.start_critic_warmup_at

        self.agent = PPO(
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
            finetune=self.warm_up_critic,
        )

    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def init_wandb(self, is_resumed_job=False):
        wandb_config = self.config.WANDB
        wandb_id = wandb.util.generate_id()
        wandb_filename = os.path.join(self.config.TENSORBOARD_DIR, "wandb_id.txt")
        wandb_resume = None
        # Reload job id if exists
        if is_resumed_job and os.path.exists(wandb_filename):
            with open(wandb_filename, "r") as file:
                wandb_id = file.read().rstrip("\n")
            wandb_resume = wandb_config.RESUME
        else:
            wandb_id=wandb.util.generate_id()
            with open(wandb_filename, 'w') as file:
                file.write(wandb_id)
        # Initialize wandb
        wandb.init(
            id=wandb_id,
            group=wandb_config.GROUP_NAME,
            project=wandb_config.PROJECT_NAME,
            config=self.config,
            mode=wandb_config.MODE,
            resume=wandb_resume,
            tags=wandb_config.TAGS,
            job_type=wandb_config.JOB_TYPE,
            dir=wandb_config.LOG_DIR,
        )
        wandb.run.name = "{}_{}".format(wandb_config.GROUP_NAME, self.config.TASK_CONFIG.SEED)
        wandb.run.save()

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats, current_episode_values
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        step_data = [a.item() for a in actions.to(device="cpu")]
        profiling_wrapper.range_pop()  # compute actions

        outputs = self.envs.step(step_data)
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        if self.use_pred_semantic and self.current_update >= self.config.MODEL.SWITCH_TO_PRED_SEMANTICS_UPDATE:
            batch["semantic"] = self.semantic_predictor(batch) # self.semantic_predictor(batch["rgb"], batch["depth"])
            # Subtract 1 from class labels for THDA YCB categories
            if self.config.MODEL.SEMANTIC_ENCODER.is_thda and self.config.MODEL.SEMANTIC_PREDICTOR.name == "rednet":
                batch["semantic"] = batch["semantic"] - 1

        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        current_episode_values += values.to(current_episode_values.device)
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward  # type: ignore
        running_episode_stats["count"] += 1 - masks  # type: ignore
        running_episode_stats["values"] += (1 - masks) * current_episode_values  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v  # type: ignore

        current_episode_reward *= masks
        current_episode_values *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, ppo_cfg, rollouts, current_episode_gae, running_episode_stats):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        values_gae = rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        prev_i = 0
        for i in range(len(values_gae)):
            current_episode_gae["sum"] += values_gae[i]
            current_episode_gae["steps"] += 1
            current_episode_gae["max"].copy_(torch.maximum(values_gae[i], current_episode_gae["max"]))
            current_episode_gae["min"].copy_(torch.minimum(values_gae[i], current_episode_gae["min"]))
            running_episode_stats["values_gae"] += (1 - rollouts.masks[i]) * current_episode_gae["sum"] # type: ignore
            running_episode_stats["values_mean"] += (1 - rollouts.masks[i]) * current_episode_gae["sum"] / current_episode_gae["steps"]  # type: ignore
            running_episode_stats["values_min"] += (1 - rollouts.masks[i]) * current_episode_gae["min"] # type: ignore
            running_episode_stats["values_max"] += (1 - rollouts.masks[i]) * current_episode_gae["max"]  # type: ignore
            # if rollouts.masks[i] == 0 and (i - prev_i) > 0:
            #     logger.info("i: {}".format(i))
            #     logger.info("pred episode ends: {},  {}, {}".format(current_episode_gae["sum"] / current_episode_gae["steps"] , current_episode_gae["max"], current_episode_gae["min"]))
            #     logger.info("episode ends: {},  {}, {}".format(sum(values_gae[prev_i:i]) / (i - prev_i +1), max(values_gae[prev_i:i]), min(values_gae[prev_i:i])))

            current_episode_gae["sum"] *= rollouts.masks[i]
            current_episode_gae["steps"] *= rollouts.masks[i]
            current_episode_gae["max"][rollouts.masks[i] == 0] = -100.0
            current_episode_gae["min"][rollouts.masks[i] == 0] = 100.0

        value_loss, action_loss, dist_entropy, avg_grad_norm = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            avg_grad_norm,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        logger.info(
            "trainable agent number of parameters: {}".format(
                sum(param.numel() if param.requires_grad else 0 for param in self.agent.parameters())
            )
        )
        self.current_update = 0

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        if self.use_pred_semantic:
            batch["semantic"] = self.semantic_predictor(batch) # self.semantic_predictor(batch["rgb"], batch["depth"])
            # Subtract 1 from class labels for THDA YCB categories
            if self.config.MODEL.SEMANTIC_ENCODER.is_thda and self.config.MODEL.SEMANTIC_PREDICTOR.name == "rednet":
                batch["semantic"] = batch["semantic"] - 1
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_values = torch.zeros(self.envs.num_envs, 1)
        current_episode_gae = dict(
            steps=torch.zeros(self.envs.num_envs, 1, device=self.device),
            sum=torch.zeros(self.envs.num_envs, 1, device=self.device),
            min=torch.ones(self.envs.num_envs, 1, device=self.device) * 100.0,
            max=torch.ones(self.envs.num_envs, 1, device=self.device) * -100.0,
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
            values=torch.zeros(self.envs.num_envs, 1),
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

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),  # type: ignore
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")
                self.current_update += 1

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                # Enable actor finetuning at update actor_finetuning_update
                if self.current_update == self.actor_finetuning_update:
                    for param in self.actor_critic.action_distribution.parameters():
                        param.requires_grad_(True)
                    for param in self.actor_critic.net.state_encoder.parameters():
                        param.requires_grad_(True)
                    logger.info("Start actor finetuning at: {}".format(self.current_update))

                    logger.info(
                        "updated agent number of parameters: {}".format(
                            sum(param.numel() if param.requires_grad else 0 for param in self.agent.parameters())
                        )
                    )

                if self.current_update > self.actor_finetuning_update:
                    lr_scheduler.step()

                profiling_wrapper.range_push("rollouts loop")
                for _step in range(ppo_cfg.num_steps):
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
                    count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                    grad_norm,
                ) = self._update_agent(ppo_cfg, rollouts, current_episode_gae, running_episode_stats)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

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
                writer.add_scalars("learning_rate", lrs, count_steps,)

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]
                losses = {k: l for l, k in zip(losses, ["value", "policy"])}
                writer.add_scalars("losses", losses, count_steps)

                writer.add_scalar("entropy", dist_entropy, count_steps)
                writer.add_scalar("grad_norm", grad_norm, count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
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

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        logger.info("Eval policy initialized")


        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
            # Subtract 1 from class labels for THDA YCB categories
            if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                batch["semantic"] = batch["semantic"] - 1

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        logger.info("Start evaluation")
        step = 0
        episode_meta = []
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes_info()

            with torch.no_grad():
                # if self.semantic_predictor is not None:
                #     batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                #     # Subtract 1 from class labels for THDA YCB categories
                #     if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                #         batch["semantic"] = batch["semantic"] - 1
                (
                    value,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                # writer.add_scalars(
                #     "eval_values",
                #     {"values_{}".format(current_episodes[0].episode_id): value},
                #     step,
                # )
                step += 1

                prev_actions.copy_(actions)  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            if self.semantic_predictor is not None:
                batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                # Subtract 1 from class labels for THDA YCB categories
                if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                    batch["semantic"] = batch["semantic"] - 1

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes_info()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    logger.info("Success: {}, SPL: {}".format(episode_stats["success"], episode_stats["spl"]))
                    episode_meta.append({
                        "scene_id": current_episodes[i].scene_id,
                        "episode_id": current_episodes[i].episode_id,
                        "metrics": episode_stats
                    })
                    write_json(episode_meta, self.config.EVAL.meta_file)
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []
                    step = 0

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    frame = append_text_to_image(frame, "Find and go to {}".format(current_episodes[i].object_category))
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        write_json(episode_meta, self.config.EVAL.meta_file)

        self.envs.close()
