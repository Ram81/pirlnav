# #!/usr/bin/env python3

# # Copyright (c) Facebook, Inc. and its affiliates.
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import contextlib
# import os
# import random
# import time
# from collections import defaultdict, deque
# from re import L
# from typing import Any, DefaultDict, Dict, List, Optional

# import numpy as np
# import torch
# import tqdm
# from habitat import Config, logger
# from habitat.utils import profiling_wrapper
# from habitat.utils.visualizations.utils import (
#     append_text_to_image,
#     observations_to_image,
# )
# from habitat_baselines.common.baseline_registry import baseline_registry
# from habitat_baselines.common.construct_vector_envs import construct_envs
# from habitat_baselines.common.environments import get_env_class
# from habitat_baselines.common.obs_transformers import (
#     apply_obs_transforms_batch,
#     apply_obs_transforms_obs_space,
#     get_active_obs_transforms,
# )
# from habitat_baselines.common.rollout_storage import RolloutStorage
# from habitat_baselines.common.tensorboard_utils import TensorboardWriter
# from habitat_baselines.rl.ddppo.algo.ddp_utils import (
#     EXIT,
#     REQUEUE,
#     SLURM_JOBID,
#     add_signal_handlers,
#     init_distrib_slurm,
#     load_resume_state,
#     requeue_job,
#     save_resume_state,
# )
# from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
# from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
# from habitat_baselines.utils.common import (
#     a,
#     batch_obs,
#     exponential_decay,
#     generate_video,
#     linear_decay,
# )
# from torch import distributed as distrib
# from torch import nn as nn
# from torch.optim.lr_scheduler import LambdaLR

# import wandb
# from pirlnav.utils.utils import critic_linear_decay, linear_warmup, setup_wandb


# @baseline_registry.register_trainer(name="pirlnav-ddppo")
# class DDPPOTrainer(PPOTrainer):
#     # DD-PPO cuts rollouts short to mitigate the straggler effect
#     # This, in theory, can cause some rollouts to be very short.
#     # All rollouts contributed equally to the loss/model-update,
#     # thus very short rollouts can be problematic.  This threshold
#     # limits the how short a short rollout can be as a fraction of the
#     # max rollout length
#     SHORT_ROLLOUT_THRESHOLD: float = 0.25

#     def __init__(self, config: Optional[Config] = None) -> None:
#         interrupted_state = load_interrupted_state()
#         self.wandb_initialized = False
#         if interrupted_state is not None:
#             config = interrupted_state["config"]

#         super().__init__(config)

#     def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
#         r"""Sets up actor critic and agent for DD-PPO.

#         Args:
#             ppo_cfg: config node with relevant params

#         Returns:
#             None
#         """
#         logger.add_filehandler(self.config.LOG_FILE)

#         policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
#         self.obs_transforms = get_active_obs_transforms(self.config)
#         observation_space = self.envs.observation_spaces[0]
#         observation_space = apply_obs_transforms_obs_space(
#             observation_space, self.obs_transforms
#         )
#         self.actor_critic = policy.from_config(
#             self.config, observation_space, self.envs.action_spaces[0]
#         )
#         self.obs_space = observation_space
#         self.actor_critic.to(self.device)

#         if (
#             self.config.RL.DDPPO.pretrained_encoder
#             or self.config.RL.DDPPO.pretrained
#         ):
#             pretrained_state = torch.load(
#                 self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
#             )
#             logger.info("Loading state")
        
#         if self.config.RL.DDPPO.pretrained:
#             missing_keys = self.actor_critic.load_state_dict(
#                 {
#                     k.replace("model.", ""): v
#                     for k, v in pretrained_state["state_dict"].items()
#                 }, strict=False
#             )
#             logger.info("Loading checkpoint missing keys: {}".format(missing_keys))

#         self.rl_finetuning = False
#         self.finetune_full_agent = False
#         self.vpt_finetuning = False
#         self.kl_coef = 0.0
#         self.pretrained_policy = None
#         self.kl_decay_coef = 0.0

#         if hasattr(self.config.RL, "Finetune") and self.config.RL.Finetune.finetune:
#             logger.info("Start Freeze encoder")
#             self.rl_finetuning = True
#             if self.config.RL.Finetune.freeze_encoders:
#                 self.actor_critic.freeze_visual_encoders()

#             self.actor_finetuning_update = self.config.RL.Finetune.start_actor_finetuning_at
#             self.actor_lr_warmup_update = self.config.RL.Finetune.actor_lr_warmup_update
#             self.critic_lr_decay_update = self.config.RL.Finetune.critic_lr_decay_update
#             self.start_critic_warmup_at = self.config.RL.Finetune.start_critic_warmup_at
#             self.finetune_full_agent =  self.config.RL.Finetune.finetune_full_agent
        
#         if hasattr(self.config.RL, "Finetune"):
#             logger.info("Start Freeze encoder")
#             if self.config.RL.Finetune.freeze_encoders:
#                 logger.info("Freeze encoders........")
#                 self.actor_critic.freeze_visual_encoders()

#         # TODO: Refactor VPT finetuning config
#         if hasattr(self.config.RL.Finetune, "vpt_finetuning") and self.config.RL.Finetune.vpt_finetuning:
#             self.actor_critic.freeze_visual_encoders()
#             logger.info("Freeze actor weights")
#             self.pretrained_policy = policy.from_config(
#                 self.config, observation_space, self.envs.action_spaces[0]
#             )
#             self.pretrained_policy.to(self.device)

#             missing_keys = self.pretrained_policy.load_state_dict(
#                 {
#                     k.replace("model.", ""): v
#                     for k, v in pretrained_state["state_dict"].items()
#                 }, strict=False
#             )

#             for param in self.pretrained_policy.parameters():
#                 param.requires_grad = False

#             self.pretrained_policy.eval()

#             self.kl_coef = self.config.RL.Finetune.kl_coef
#             self.kl_decay_coef = self.config.RL.Finetune.kl_decay_coef
#             self.vpt_finetuning = self.config.RL.Finetune.vpt_finetuning

#             if self.config.RL.Finetune.zero_critic_weights:
#                 logger.info("Initialize critic weights to zero")
#                 self.actor_critic.critic.fc.weight.data.fill_(0.0)

#         if self.config.RL.DDPPO.reset_critic:
#             nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
#             nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

#         self.agent = DDPPO(
#             actor_critic=self.actor_critic,
#             clip_param=ppo_cfg.clip_param,
#             ppo_epoch=ppo_cfg.ppo_epoch,
#             num_mini_batch=ppo_cfg.num_mini_batch,
#             value_loss_coef=ppo_cfg.value_loss_coef,
#             entropy_coef=ppo_cfg.entropy_coef,
#             lr=ppo_cfg.lr,
#             eps=ppo_cfg.eps,
#             max_grad_norm=ppo_cfg.max_grad_norm,
#             use_normalized_advantage=ppo_cfg.use_normalized_advantage,
#             finetune=self.rl_finetuning,
#             finetune_full_agent=self.finetune_full_agent,
#             vpt_finetuning=self.vpt_finetuning,
#             kl_coef=self.kl_coef,
#             pretrained_policy=self.pretrained_policy
#         )

#     @profiling_wrapper.RangeContext("train")
#     def train(self) -> None:
#         r"""Main method for DD-PPO.

#         Returns:
#             None
#         """
#         self.local_rank, tcp_store = init_distrib_slurm(
#             self.config.RL.DDPPO.distrib_backend
#         )
#         add_signal_handlers()

#         profiling_wrapper.configure(
#             capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
#             num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
#         )

#         interrupted_state = None

#         # Check resume state file in config
#         resume_state_file = self.config.RESUME_STATE_FILE
#         if resume_state_file is not None and os.path.exists(resume_state_file):
#             interrupted_state = load_interrupted_state(resume_state_file)

#         interrupted_state_file = os.path.join(self.config.CHECKPOINT_FOLDER, "{}.pth".format(SLURM_JOBID))
#         # Override resume state file if the current job interrupted state exists
#         if interrupted_state_file is not None and os.path.exists(interrupted_state_file):
#             interrupted_state = load_interrupted_state(interrupted_state_file)

#         if interrupted_state is not None:
#             logger.info("Overriding current config with interrupted state config")
#             self.config = interrupted_state["config"]

#         # Stores the number of workers that have finished their rollout
#         num_rollouts_done_store = distrib.PrefixStore(
#             "rollout_tracker", tcp_store
#         )
#         num_rollouts_done_store.set("num_done", "0")

#         self.world_rank = distrib.get_rank()
#         self.world_size = distrib.get_world_size()
#         self.current_update = 0

#         self.config.defrost()
#         self.config.TORCH_GPU_ID = self.local_rank
#         self.config.SIMULATOR_GPU_ID = self.local_rank
#         # Multiply by the number of simulators to make sure they also get unique seeds
#         self.config.TASK_CONFIG.SEED += (
#             self.world_rank * self.config.NUM_PROCESSES
#         )
#         self.config.freeze()

#         random.seed(self.config.TASK_CONFIG.SEED)
#         np.random.seed(self.config.TASK_CONFIG.SEED)
#         torch.manual_seed(self.config.TASK_CONFIG.SEED)

#         if torch.cuda.is_available():
#             self.device = torch.device("cuda", self.local_rank)
#             torch.cuda.set_device(self.device)
#         else:
#             self.device = torch.device("cpu")

#         self.envs = construct_envs(
#             self.config,
#             get_env_class(self.config.ENV_NAME),
#             workers_ignore_signals=True,
#         )

#         ppo_cfg = self.config.RL.PPO
#         if (
#             not os.path.isdir(self.config.CHECKPOINT_FOLDER)
#             and self.world_rank == 0
#         ):
#             os.makedirs(self.config.CHECKPOINT_FOLDER)

#         self._setup_actor_critic_agent(ppo_cfg)
#         self.agent.init_distributed(find_unused_params=True)

#         if self.world_rank == 0:
#             parameters = [(param.numel(), param.requires_grad) for param in self.agent.parameters()] 
#             logger.info(
#                 "agent number of trainable parameters: {} / {}".format(
#                     sum(
#                         param[0]
#                         for param in parameters
#                         if param[1]
#                     ),
#                     sum(
#                         param[0]
#                         for param in parameters
#                     )
#                 )
#             )

#             if self.wandb_initialized == False:
#                 setup_wandb(self.config, train=True, project_name=self.config.WANDB_PROJECT_NAME)
#                 self.wandb_initialized = True


#         observations = self.envs.reset()
#         batch = batch_obs(observations, device=self.device)

#         batch = apply_obs_transforms_batch(batch, self.obs_transforms)

#         obs_space = self.obs_space

#         rollouts = RolloutStorage(
#             ppo_cfg.num_steps,
#             self.envs.num_envs,
#             obs_space,
#             self.envs.action_spaces[0],
#             self.config.MODEL.STATE_ENCODER.hidden_size,
#             num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
#         )
#         rollouts.to(self.device)

#         for sensor in rollouts.observations:
#             rollouts.observations[sensor][0].copy_(batch[sensor])

#         # batch and observations may contain shared PyTorch CUDA
#         # tensors.  We must explicitly clear them here otherwise
#         # they will be kept in memory for the entire duration of training!
#         batch = None
#         observations = None

#         current_episode_reward = torch.zeros(
#             self.envs.num_envs, 1, device=self.device
#         )
#         current_episode_values = torch.zeros(
#             self.envs.num_envs, 1, device=self.device
#         )
#         current_episode_gae = dict(
#             steps=torch.zeros(self.envs.num_envs, 1, device=self.device),
#             sum=torch.zeros(self.envs.num_envs, 1, device=self.device),
#             min=torch.ones(self.envs.num_envs, 1, device=self.device) * 100.0,
#             max=torch.ones(self.envs.num_envs, 1, device=self.device) * -100.0,
#         )
#         running_episode_stats = dict(
#             count=torch.zeros(self.envs.num_envs, 1, device=self.device),
#             reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
#             values=torch.zeros(self.envs.num_envs, 1, device=self.device),
#             values_gae=torch.zeros(self.envs.num_envs, 1, device=self.device), 
#             values_mean=torch.zeros(self.envs.num_envs, 1, device=self.device),
#             values_min=torch.zeros(self.envs.num_envs, 1, device=self.device), 
#             values_max=torch.zeros(self.envs.num_envs, 1, device=self.device), 
#         )
#         window_episode_stats: DefaultDict[str, deque] = defaultdict(
#             lambda: deque(maxlen=ppo_cfg.reward_window_size)
#         )

#         t_start = time.time()
#         env_time = 0
#         pth_time = 0
#         count_steps = 0
#         count_checkpoints = 0
#         start_update = 0
#         prev_time = 0

#         if self.rl_finetuning:
#             if self.finetune_full_agent:
#                 lr_scheduler = LambdaLR(
#                     optimizer=self.agent.optimizer,
#                     lr_lambda=[
#                         lambda x: critic_linear_decay(x, self.start_critic_warmup_at, self.critic_lr_decay_update, self.config.RL.PPO.lr, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                     ]
#                 ) 
#             else:   
#                 lr_scheduler = LambdaLR(
#                     optimizer=self.agent.optimizer,
#                     lr_lambda=[
#                         lambda x: critic_linear_decay(x, self.start_critic_warmup_at, self.critic_lr_decay_update, self.config.RL.PPO.lr, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                         lambda x: linear_warmup(x, self.actor_finetuning_update, self.actor_lr_warmup_update, 0.0, self.config.RL.Finetune.policy_ft_lr),
#                     ]
#                 )
#         else:
#             lr_scheduler = LambdaLR(
#                 optimizer=self.agent.optimizer,
#                 lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
#             )

#         if interrupted_state is not None:
#             self.agent.load_state_dict(interrupted_state["state_dict"])
#             self.agent.optimizer.load_state_dict(
#                 interrupted_state["optim_state"]
#             )
#             lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

#             requeue_stats = interrupted_state["requeue_stats"]
#             env_time = requeue_stats["env_time"]
#             pth_time = requeue_stats["pth_time"]
#             count_steps = requeue_stats["count_steps"]
#             count_checkpoints = requeue_stats["count_checkpoints"]
#             start_update = requeue_stats["start_update"]
#             prev_time = requeue_stats["prev_time"]
#             self.current_update = start_update

#             if self.rl_finetuning and self.current_update >= self.actor_finetuning_update:
#                 for param in self.actor_critic.action_distribution.parameters():
#                     param.requires_grad_(True)
#                 for param in self.actor_critic.net.state_encoder.parameters():
#                     param.requires_grad_(True)      
#                 logger.info("unfreezing params")
#             logger.info("Loaded state from interrupted state")
        
#         logger.info("Current update: {}, Max updates: {}".format(self.current_update, self.config.NUM_UPDATES))

#         with (
#             TensorboardWriter(
#                 self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
#             )
#             if self.world_rank == 0
#             else contextlib.suppress()
#         ) as writer:
#             for update in range(start_update, self.config.NUM_UPDATES):
#                 profiling_wrapper.on_start_step()
#                 profiling_wrapper.range_push("train update")
#                 self.current_update += 1

#                 if ppo_cfg.use_linear_lr_decay:
#                     lr_scheduler.step()  # type: ignore

#                 if ppo_cfg.use_linear_clip_decay:
#                     self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
#                         update, self.config.NUM_UPDATES
#                     )
                
#                 if self.kl_decay_coef > 0:
#                     self.agent.kl_coef = exponential_decay(self.agent.kl_coef, self.kl_decay_coef)

#                 if EXIT.is_set():
#                     profiling_wrapper.range_pop()  # train update

#                     self.envs.close()

#                     if self.world_rank == 0:
#                         requeue_stats = dict(
#                             env_time=env_time,
#                             pth_time=pth_time,
#                             count_steps=count_steps,
#                             count_checkpoints=count_checkpoints,
#                             start_update=update,
#                             prev_time=(time.time() - t_start) + prev_time,
#                         )
#                         logger.info("save interrupted state at: {}".format(interrupted_state_file))
#                         save_interrupted_state(
#                             dict(
#                                 state_dict=self.agent.state_dict(),
#                                 optim_state=self.agent.optimizer.state_dict(),
#                                 lr_sched_state=lr_scheduler.state_dict(),
#                                 config=self.config,
#                                 requeue_stats=requeue_stats,
#                             ),
#                             interrupted_state_file
#                         )

#                     requeue_job()
#                     return

#                 # Enable actor finetuning at update actor_finetuning_update
#                 if self.rl_finetuning and self.current_update == self.actor_finetuning_update:
#                     for param in self.actor_critic.action_distribution.parameters():
#                         param.requires_grad_(True)
#                     for param in self.actor_critic.net.state_encoder.parameters():
#                         param.requires_grad_(True)
#                     # Unfreeze visual encoder when RL-FT
#                     if self.config.RL.Finetune.unfreeze_encoders_after_warmup:
#                         self.actor_critic.unfreeze_visual_encoders()

#                     for i, param_group in enumerate(self.agent.optimizer.param_groups):
#                         param_group["eps"] = self.config.RL.PPO.eps
#                         lr_scheduler.base_lrs[i] = 1.0

#                     if self.world_rank == 0:
#                         logger.info("Start actor finetuning at: {}".format(self.current_update))

#                         logger.info(
#                             "updated agent number of parameters: {}".format(
#                                 sum(param.numel() if param.requires_grad else 0 for param in self.agent.parameters())
#                             )
#                         )
#                 if self.rl_finetuning and self.current_update == self.start_critic_warmup_at:
#                     self.agent.optimizer.param_groups[0]["eps"] = self.config.RL.PPO.eps
#                     lr_scheduler.base_lrs[0] = 1.0
#                     if self.world_rank == 0:
#                         logger.info("Set critic LR at: {}".format(self.current_update))


#                 count_steps_delta = 0
#                 self.agent.eval()
#                 profiling_wrapper.range_push("rollouts loop")
#                 for step in range(ppo_cfg.num_steps):

#                     (
#                         delta_pth_time,
#                         delta_env_time,
#                         delta_steps,
#                     ) = self._collect_rollout_step(
#                         rollouts, current_episode_reward,
#                         running_episode_stats, current_episode_values
#                     )
#                     pth_time += delta_pth_time
#                     env_time += delta_env_time
#                     count_steps_delta += delta_steps

#                     # This is where the preemption of workers happens.  If a
#                     # worker detects it will be a straggler, it preempts itself!
#                     if (
#                         step
#                         >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
#                     ) and int(num_rollouts_done_store.get("num_done")) > (
#                         self.config.RL.DDPPO.sync_frac * self.world_size
#                     ):
#                         break
#                 profiling_wrapper.range_pop()  # rollouts loop

#                 num_rollouts_done_store.add("num_done", 1)

#                 self.agent.train()

#                 (
#                     delta_pth_time,
#                     value_loss,
#                     action_loss,
#                     dist_entropy,
#                     grad_norm,
#                     aux_kl_constraint_epoch,
#                 ) = self._update_agent(ppo_cfg, rollouts, current_episode_gae, running_episode_stats)
#                 pth_time += delta_pth_time

#                 stats_ordering = sorted(running_episode_stats.keys())
#                 stats = torch.stack(
#                     [running_episode_stats[k] for k in stats_ordering], 0
#                 )
#                 distrib.all_reduce(stats)

#                 for i, k in enumerate(stats_ordering):
#                     window_episode_stats[k].append(stats[i].clone())

#                 stats = torch.tensor(
#                     [value_loss, action_loss, count_steps_delta, dist_entropy, grad_norm, aux_kl_constraint_epoch],
#                     device=self.device,
#                 )
#                 distrib.all_reduce(stats)
#                 count_steps += stats[2].long().item()

#                 if self.world_rank == 0:
#                     num_rollouts_done_store.set("num_done", "0")

#                     losses = [
#                         stats[0].item() / self.world_size,
#                         stats[1].item() / self.world_size,
                        
#                     ]
#                     deltas = {
#                         k: (
#                             (v[-1] - v[0]).sum().item()
#                             if len(v) > 1
#                             else v[0].sum().item()
#                         )
#                         for k, v in window_episode_stats.items()
#                     }
#                     deltas["count"] = max(deltas["count"], 1.0)

#                     lrs = {}
#                     for i, param_group in enumerate(self.agent.optimizer.param_groups):
#                         lrs["pg_{}".format(i)] = param_group["lr"]

#                     wandb.log({"reward": deltas["reward"] / deltas["count"]}, step=count_steps)
#                     wandb.log({"learning_rate": lrs}, step=count_steps)
#                     wandb.log({"kl_coef": self.agent.kl_coef}, step=count_steps)

#                     # Check to see if there are any metrics
#                     # that haven't been logged yet
#                     metrics = {
#                         k: v / deltas["count"]
#                         for k, v in deltas.items()
#                         if k not in {"reward", "count"}
#                     }
#                     if len(metrics) > 0:
#                         wandb.log(metrics, step=count_steps)

#                     losses = {k: l for l, k in zip(losses, ["value", "policy"])}
#                     wandb.log(losses, step=count_steps)

#                     wandb.log({"entropy": stats[3].item() / self.world_size}, step=count_steps)
#                     wandb.log({"grad_norm", stats[4].item() / self.world_size}, step=count_steps)
#                     wandb.log({"kl_constraint", stats[5].item() / self.world_size}, step=count_steps)

#                     # log stats
#                     if update > 0 and update % self.config.LOG_INTERVAL == 0:
#                         logger.info(
#                             "update: {}\tfps: {:.3f}\t".format(
#                                 update,
#                                 count_steps
#                                 / ((time.time() - t_start) + prev_time),
#                             )
#                         )

#                         logger.info(
#                             "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
#                             "frames: {}".format(
#                                 update, env_time, pth_time, count_steps
#                             )
#                         )
#                         logger.info(
#                             "Average window size: {}  {}".format(
#                                 len(window_episode_stats["count"]),
#                                 "  ".join(
#                                     "{}: {:.3f}".format(k, v / deltas["count"])
#                                     for k, v in deltas.items()
#                                     if k != "count"
#                                 ),
#                             )
#                         )
#                         logger.info(
#                             "update: {}\tLR: {}\tPG_LR: {}".format(
#                                 update,
#                                 lr_scheduler.get_lr(),
#                                 [param_group["lr"] for param_group in self.agent.optimizer.param_groups],
#                             )
#                         )

#                     # checkpoint model
#                     if update % self.config.CHECKPOINT_INTERVAL == 0:
#                         self.save_checkpoint(
#                             f"ckpt.{count_checkpoints}.pth",
#                             dict(step=count_steps),
#                         )
#                         count_checkpoints += 1

#                 profiling_wrapper.range_pop()  # train update

#             self.envs.close()

#             if self.world_rank == 0:
#                 requeue_stats = dict(
#                     env_time=env_time,
#                     pth_time=pth_time,
#                     count_steps=count_steps,
#                     count_checkpoints=count_checkpoints,
#                     start_update=update,
#                     prev_time=(time.time() - t_start) + prev_time,
#                 )
#                 logger.info("save interrupted state at end: {}".format(interrupted_state_file))
#                 save_interrupted_state(
#                     dict(
#                         state_dict=self.agent.state_dict(),
#                         optim_state=self.agent.optimizer.state_dict(),
#                         lr_sched_state=lr_scheduler.state_dict(),
#                         config=self.config,
#                         requeue_stats=requeue_stats,
#                     ),
#                     interrupted_state_file
#                 )

#             requeue_job()
#             return
    
#     def _eval_checkpoint(
#         self,
#         checkpoint_path: str,
#         writer: TensorboardWriter,
#         checkpoint_index: int = 0,
#     ) -> None:
#         r"""Evaluates a single checkpoint.

#         Args:
#             checkpoint_path: path of checkpoint
#             writer: tensorboard writer object for logging to tensorboard
#             checkpoint_index: index of cur checkpoint for logging

#         Returns:
#             None
#         """
#         # Map location CPU is almost always better than mapping to a CUDA device.
#         ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

#         if self.config.EVAL.USE_CKPT_CONFIG:
#             config = self._setup_eval_config(ckpt_dict["config"])
#         else:
#             config = self.config.clone()

#         ppo_cfg = config.RL.PPO

#         config.defrost()
#         config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
#         config.freeze()

#         if len(self.config.VIDEO_OPTION) > 0:
#             config.defrost()
#             config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#             config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
#             config.freeze()

#         logger.info(f"env config: {config}")
#         self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
#         self._setup_actor_critic_agent(ppo_cfg)

#         self.agent.load_state_dict(ckpt_dict["state_dict"])
#         self.actor_critic = self.agent.actor_critic
#         logger.info("Eval policy initialized")

#         if self.wandb_initialized == False:
#             setup_wandb(self.config, train=False, project_name=self.config.WANDB_PROJECT_NAME)
#             self.wandb_initialized = True

#         observations = self.envs.reset()
#         batch = batch_obs(observations, device=self.device)
#         batch = apply_obs_transforms_batch(batch, self.obs_transforms)

#         current_episode_reward = torch.zeros(
#             self.envs.num_envs, 1, device=self.device
#         )
#         test_recurrent_hidden_states = torch.zeros(
#             self.actor_critic.net.num_recurrent_layers,
#             self.config.NUM_PROCESSES,
#             ppo_cfg.hidden_size,
#             device=self.device,
#         )
#         prev_actions = torch.zeros(
#             self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
#         )
#         not_done_masks = torch.zeros(
#             self.config.NUM_PROCESSES, 1, device=self.device
#         )
#         stats_episodes: Dict[
#             Any, Any
#         ] = {}  # dict of dicts that stores stats per episode

#         rgb_frames = [
#             [] for _ in range(self.config.NUM_PROCESSES)
#         ]  # type: List[List[np.ndarray]]
#         if len(self.config.VIDEO_OPTION) > 0:
#             os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

#         number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
#         if number_of_eval_episodes == -1:
#             number_of_eval_episodes = sum(self.envs.number_of_episodes)
#         else:
#             total_num_eps = sum(self.envs.number_of_episodes)
#             if total_num_eps < number_of_eval_episodes:
#                 logger.warn(
#                     f"Config specified {number_of_eval_episodes} eval episodes"
#                     ", dataset only has {total_num_eps}."
#                 )
#                 logger.warn(f"Evaluating with {total_num_eps} instead.")
#                 number_of_eval_episodes = total_num_eps

#         pbar = tqdm.tqdm(total=number_of_eval_episodes)
#         self.actor_critic.eval()
#         logger.info("Start evaluation")
#         step = 0

#         while (
#             len(stats_episodes) < number_of_eval_episodes
#             and self.envs.num_envs > 0
#         ):
#             current_episodes = self.envs.current_episodes_info()

#             with torch.no_grad():
#                 (
#                     value,
#                     actions,
#                     _,
#                     test_recurrent_hidden_states,
#                     dist_entropy,
#                 ) = self.actor_critic.act(
#                     batch,
#                     test_recurrent_hidden_states,
#                     prev_actions,
#                     not_done_masks,
#                     deterministic=False,
#                 )
#                 step += 1

#                 prev_actions.copy_(actions)  # type: ignore

#             # NB: Move actions to CPU.  If CUDA tensors are
#             # sent in to env.step(), that will create CUDA contexts
#             # in the subprocesses.
#             # For backwards compatibility, we also call .item() to convert to
#             # an int
#             step_data = [a.item() for a in actions.to(device="cpu")]

#             outputs = self.envs.step(step_data)

#             observations, rewards_l, dones, infos = [
#                 list(x) for x in zip(*outputs)
#             ]
#             batch = batch_obs(observations, device=self.device)
#             batch = apply_obs_transforms_batch(batch, self.obs_transforms)

#             not_done_masks = torch.tensor(
#                 [[0.0] if done else [1.0] for done in dones],
#                 dtype=torch.float,
#                 device=self.device,
#             )

#             rewards = torch.tensor(
#                 rewards_l, dtype=torch.float, device=self.device
#             ).unsqueeze(1)

#             current_episode_reward += rewards

#             next_episodes = self.envs.current_episodes_info()
#             envs_to_pause = []
#             n_envs = self.envs.num_envs
#             for i in range(n_envs):
#                 if (
#                     next_episodes[i].scene_id,
#                     next_episodes[i].episode_id,
#                 ) in stats_episodes:
#                     envs_to_pause.append(i)

#                 # episode ended
#                 if not_done_masks[i].item() == 0:
#                     pbar.update()
#                     episode_stats = {}
#                     episode_stats["reward"] = current_episode_reward[i].item()
#                     episode_stats.update(
#                         self._extract_scalars_from_info(infos[i])
#                     )
#                     current_episode_reward[i] = 0
#                     logger.info("Success: {}, SPL: {}".format(episode_stats["success"], episode_stats["spl"]))

#                     # use scene_id + episode_id as unique id for storing stats
#                     stats_episodes[
#                         (
#                             current_episodes[i].scene_id,
#                             current_episodes[i].episode_id,
#                         )
#                     ] = episode_stats

#                     if len(self.config.VIDEO_OPTION) > 0:
#                         generate_video(
#                             video_option=self.config.VIDEO_OPTION,
#                             video_dir=self.config.VIDEO_DIR,
#                             images=rgb_frames[i],
#                             episode_id=current_episodes[i].episode_id,
#                             checkpoint_idx=checkpoint_index,
#                             metrics={"success": episode_stats["success"], "spl": episode_stats["spl"]},
#                             tb_writer=writer,
#                         )

#                         rgb_frames[i] = []
#                     step = 0

#                 # episode continues
#                 elif len(self.config.VIDEO_OPTION) > 0:
#                     # TODO move normalization / channel changing out of the policy and undo it here
#                     frame = observations_to_image(
#                         {k: v[i] for k, v in batch.items()}, infos[i]
#                     )
#                     frame = append_text_to_image(frame, "Find and go to {}".format(current_episodes[i].object_category))
#                     rgb_frames[i].append(frame)

#             (
#                 self.envs,
#                 test_recurrent_hidden_states,
#                 not_done_masks,
#                 current_episode_reward,
#                 prev_actions,
#                 batch,
#                 rgb_frames,
#             ) = self._pause_envs(
#                 envs_to_pause,
#                 self.envs,
#                 test_recurrent_hidden_states,
#                 not_done_masks,
#                 current_episode_reward,
#                 prev_actions,
#                 batch,
#                 rgb_frames,
#             )

#         num_episodes = len(stats_episodes)
#         aggregated_stats = {}
#         for stat_key in next(iter(stats_episodes.values())).keys():
#             aggregated_stats[stat_key] = (
#                 sum(v[stat_key] for v in stats_episodes.values())
#                 / num_episodes
#             )

#         for k, v in aggregated_stats.items():
#             logger.info(f"Average episode {k}: {v:.4f}")
#         logger.info("Checkpoint path: {}".format(checkpoint_path))

#         step_id = checkpoint_index
#         if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
#             step_id = ckpt_dict["extra_state"]["step"]

#         wandb.log({"average reward": aggregated_stats["reward"]}, count=step_id)

#         metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
#         if len(metrics) > 0:
#             wandb.log(metrics, step_id)

#         self.envs.close()
