#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from collections import defaultdict
from logging import Logger, log
from typing import Optional, Type

import numpy as np
import habitat
from habitat import Config, Dataset, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.pickplace.pickplace import (
    AgentToReceptacleDistance,
    ObjectToReceptacleDistance,
    AgentToObjectDistance
)


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="RearrangementRLEnv")
class RearrangementRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD
        # Get reward from RearrangementReward measure
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward += current_measure
        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]
    

    def _end_early(self):
        end_early = 0
        end_early += self._env.get_metrics()[AgentToObjectDistance.cls_uuid]["was_nan"]
        end_early += self._env.get_metrics()[AgentToReceptacleDistance.cls_uuid]["was_nan"]
        end_early += self._env.get_metrics()[ObjectToReceptacleDistance.cls_uuid]["was_nan"]
        return end_early

    def get_done(self, observations):
        done = False
        if self._end_early():
            logger.info("End early")

        if self._env.episode_over or self._episode_success() or self._end_early():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="ExploreNavRLEnv")
class ExploreNavRLEnv(NavRLEnv):
    r"""
        We want to train an agent that overfits less. We provide an exploration reward and a delayed gratification success reward.
        This is to avoid weird shaping loss.
        We provide quickly attenuating coverage reward, and a gradually increasing success reward.
    """
    def __init__(self, config, dataset=None): # add coverage to the metrics
        super().__init__(config, dataset)
        self.step_penalty = 1

    def step(self, *args, **kwargs):
        self.step_penalty *= self._rl_config.COVERAGE_ATTENUATION
        return super().step(*args, **kwargs)

    def reset(self):
        self.step_penalty = 1
        return super().reset()

    def get_reward_range(self):
        old_low, old_hi = super().get_reward_range()
        return old_low, old_hi + self._rl_config.COVERAGE_REWARD

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD
        visit = self._env.get_metrics()["coverage"]["visit_count"]
        reward += self.step_penalty * self._rl_config.COVERAGE_REWARD / (visit ** self._rl_config.COVERAGE_VISIT_EXP)
        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        return reward


@baseline_registry.register_env(name="ObjectNavRLEnv")
class ObjectNavRLEnv(NavRLEnv):
    r"""
        We want to train an agent that overfits less. We provide an exploration reward and a delayed gratification success reward.
        This is to avoid weird shaping loss.
        We provide quickly attenuating coverage reward, and a gradually increasing success reward.
    """
    def __init__(self, config, dataset=None): # add coverage to the metrics
        super().__init__(config, dataset)

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def reset(self):
        return super().reset()

    def get_reward_range(self):
        old_low, old_hi = super().get_reward_range()
        return old_low, old_hi

    def get_reward(self, observations):
        reward = 0
        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        return reward


@baseline_registry.register_env(name="ObjectNavDenseRewardEnv")
class ObjectNavDenseRewardEnv(NavRLEnv):
    r"""
        ObjectNav RL Env with dense reward measure
    """
    def __init__(self, config, dataset=None): # add coverage to the metrics
        super().__init__(config, dataset)
        self.config = config

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def reset(self):
        return super().reset()

    def get_reward_range(self):
        old_low, old_hi = super().get_reward_range()
        return old_low, old_hi

    def get_reward(self, observations):
        reward = self._env.get_metrics()[self.config.RL.REWARD_MEASURE]
        return reward
