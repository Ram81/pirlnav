#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import random
import time
import torch
import tqdm

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor
from numpy import ndarray
from torch.utils.data import DataLoader, ConcatDataset

from habitat import logger, make_dataset
from habitat.core.env import Env, RLEnv
from habitat.core.vector_env import VectorEnv
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image, append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.disk_based.dataset.dataset import PickPlaceDataset, collate_fn
from habitat_baselines.il.disk_based.policy.resnet_policy import PickPlacePolicy
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
)


@baseline_registry.register_trainer(name="rearrangement-behavior-cloning")
class RearrangementBCTrainer(BaseILTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["PickPlaceTask-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        self.obs_transforms = []

        if config is not None:
            logger.info(f"config: {config}")

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _make_results_dir(self, split="val"):
        r"""Makes directory for saving eqa-cnn-pretrain eval results."""
        for s_type in ["rgb", "seg", "depth", "top_down_map"]:
            dir_name = self.config.RESULTS_DIR.format(split=split, type=s_type)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "goal_vis_pixels", "rearrangement_reward", "coverage"}

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
    
    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[VectorEnv, RLEnv, Env],
        test_recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
    ) -> Tuple[
        Union[VectorEnv, RLEnv, Env],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]

            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _setup_actor_critic_agent(self, observation_space, action_space, model_config):
        model_config.defrost()
        model_config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_config.freeze()

        policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        policy = policy.from_config(
            self.config, observation_space, action_space
        )
        return policy

    def _setup_dataset(self):
        config = self.config

        content_scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
        if "*" in content_scenes:
            dataset = make_dataset(
                config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
            )
            content_scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
        datasets = []
        print(content_scenes)
        for scene in ["q9vSo1VnCiC"]:
            dataset = PickPlaceDataset(
                config,
                content_scenes=[scene],
                use_iw=config.IL.USE_IW,
                inflection_weight_coef=config.MODEL.inflection_weight_coef
            )
            datasets.append(dataset)

        concat_dataset = ConcatDataset(datasets)
        return concat_dataset


    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        config = self.config

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=True
        )

        rearrangement_dataset = self._setup_dataset()
        batch_size = config.IL.BehaviorCloning.batch_size

        logger.info("Setup dataloader")

        train_loader = DataLoader(
            rearrangement_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        logger.info("Dataloader setup")

        logger.info(
            "[ train_loader has {} samples ]".format(
                len(rearrangement_dataset)
            )
        )

        action_space = self.envs.action_spaces[0]

        self.policy = self._setup_actor_critic_agent(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL
        )
        self.policy = torch.nn.DataParallel(self.policy, dim=1)
        self.policy.to(self.device)

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=float(config.IL.BehaviorCloning.lr),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=config.IL.BehaviorCloning.lr,
            steps_per_epoch=len(train_loader), epochs=config.IL.BehaviorCloning.max_epochs
        )

        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        epoch, t = 1, 0
        logger.info("Starting training: {}".format(self.device))
        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
        ) as writer:
            while epoch <= config.IL.BehaviorCloning.max_epochs:
                logger.info("Epoch: {}".format(epoch))
                start_time = time.time()
                avg_loss = 0.0
                avg_load_time = 0.0
                avg_slice_time = 0.0
                avg_train_time = 0.0

                batch_start_time = time.time()

                for batch in train_loader:
                    torch.cuda.empty_cache()
                    t += 1

                    (
                     observations_batch,
                     gt_prev_action,
                     episode_not_done,
                     gt_next_action,
                     inflec_weights
                    ) = batch

                    avg_load_time += ((time.time() - batch_start_time) / 60)

                    rnn_hidden_states = torch.zeros(
                        config.MODEL.STATE_ENCODER.num_recurrent_layers,
                        gt_prev_action.shape[1],
                        config.MODEL.STATE_ENCODER.hidden_size,
                        device=self.device,
                    )

                    optim.zero_grad()

                    num_samples = gt_prev_action.shape[0]
                    timestep_batch_size = config.IL.BehaviorCloning.timestep_batch_size
                    num_steps = num_samples // timestep_batch_size + (num_samples % timestep_batch_size != 0)
                    batch_loss = 0
                    for i in range(num_steps):
                        slice_start_time = time.time()
                        start_idx = i * timestep_batch_size
                        end_idx = start_idx + timestep_batch_size
                        observations_batch_sample = {
                            k: v[start_idx:end_idx].to(device=self.device)
                            for k, v in observations_batch.items()
                        }

                        gt_next_action_sample = gt_next_action[start_idx:end_idx].long().to(self.device)
                        gt_prev_action_sample = gt_prev_action[start_idx:end_idx].long().to(self.device)
                        episode_not_dones_sample = episode_not_done[start_idx:end_idx].long().to(self.device)
                        inflec_weights_sample = inflec_weights[start_idx:end_idx].long().to(self.device)

                        avg_slice_time += ((time.time() - slice_start_time) / 60)

                        train_time = time.time()

                        logits, rnn_hidden_states = self.policy(
                            observations_batch_sample,
                            rnn_hidden_states,
                            gt_prev_action_sample,
                            episode_not_dones_sample
                        )

                        T, N = gt_next_action_sample.shape
                        logits = logits.view(T, N, -1)

                        action_loss = cross_entropy_loss(logits.permute(0, 2, 1), gt_next_action_sample)
                        denom = inflec_weights_sample.sum(0)
                        denom[denom == 0.0] = 1
                        action_loss = ((inflec_weights_sample * action_loss).sum(0) / denom).mean()
                        loss = (action_loss / num_steps)
                        loss.backward()
                        batch_loss += loss.item()
                        rnn_hidden_states = rnn_hidden_states.detach()

                        avg_train_time += ((time.time() - train_time) / 60)

                    if t % config.LOG_INTERVAL == 0:
                        logger.info(
                            "[ Epoch: {}; iter: {}; loss: {:.3f}; load time: {:.3f}; train time: {:.3f};]".format(
                                epoch, t, batch_loss, avg_load_time / t, avg_train_time / t,
                            )
                        )

                    optim.step()
                    scheduler.step()
                    batch_start_time = time.time()
                    avg_loss += batch_loss

                end_time = time.time()
                time_taken = "{:.1f}".format((end_time - start_time) / 60)
                avg_loss = avg_loss / len(train_loader)
                avg_train_time = avg_train_time / len(train_loader)
                avg_load_time = avg_load_time / len(train_loader)

                logger.info(
                    "[ Epoch {} completed. Time taken: {} minutes. Load time: {} mins. Train time: {} mins]".format(
                        epoch, time_taken, avg_load_time, avg_train_time
                    )
                )
                logger.info("[ Average loss: {:.3f} ]".format(avg_loss))
                writer.add_scalar("avg_train_loss", avg_loss, epoch)

                print("-----------------------------------------")

                if epoch % config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        self.policy.module.state_dict(), "model_{}.ckpt".format(epoch)
                    )

                epoch += 1
        logger.info("Epochs ended")
        self.envs.close()
        logger.info("Closing envs")

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
        config = self.config

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            if config.SHOW_TOP_DOWN_MAP:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        batch_size = config.IL.BehaviorCloning.batch_size

        logger.info(
            "[ val_loader has {} samples ]".format(
                self.envs.count_episodes()
            )
        )

        action_space = self.envs.action_spaces[0]

        self.policy = self._setup_actor_critic_agent(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL
        )

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(ckpt_dict, strict=True)
        self.policy.to(self.device)
        self.policy.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        prev_actions = torch.zeros(
            self.envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        rnn_hidden_states = torch.zeros(
            config.MODEL.STATE_ENCODER.num_recurrent_layers,
            self.envs.num_envs,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
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
        action_indices = [1] * self.envs.num_envs
        action_match_count = [(0, 0)] * self.envs.num_envs
        possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        episode_count = 0
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    logits,
                    rnn_hidden_states
                ) = self.policy(
                    batch,
                    rnn_hidden_states,
                    prev_actions,
                    not_done_masks
                )

                actions = torch.argmax(logits, dim=1)
                prev_actions.copy_(actions.unsqueeze(1))

            action_names = [possible_actions[a.item()] for a in actions.to(device="cpu")]
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

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
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
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    next_episodes = self.envs.current_episodes()
                    episode_count += 1

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=episode_count,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []
                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    frame = append_text_to_image(frame, "Instruction: {}".format(next_episodes[i].instruction.instruction_text))
                    rgb_frames[i].append(frame)


            (
                self.envs,
                rnn_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                rnn_hidden_states,
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

        self.envs.close()

        print ("environments closed")
