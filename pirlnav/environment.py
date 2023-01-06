from typing import Optional

import habitat
import numpy as np
from habitat import Config, Dataset


@habitat.registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]:
            return True
        return False

    def get_info(self, observations):
        return self._env.get_metrics()


# @habitat.registry.register_env(name="SimpleRLEnv")
# class NavRLEnv(habitat.RLEnv):
#     def __init__(self, config: Config, dataset: Optional[Dataset] = None):
#         super().__init__(config, dataset)
#         self._reward_measure_name = self.config.TASK.REWARD_MEASURE
#         self._success_measure_name = self.config.TASK.SUCCESS_MEASURE

#         self._previous_measure: Optional[float] = None

#     def reset(self):
#         observations = super().reset()
#         self._previous_measure = self._env.get_metrics()[
#             self._reward_measure_name
#         ]
#         return observations

#     def step(self, *args, **kwargs):
#         return super().step(*args, **kwargs)

#     def get_reward_range(self):
#         return (
#             self.config.TASK.SLACK_REWARD - 1.0,
#             self.config.TASK.SUCCESS_REWARD + 1.0,
#         )

#     def get_reward(self, observations):
#         reward = self.config.TASK.SLACK_REWARD

#         current_measure = self._env.get_metrics()[self._reward_measure_name]

#         reward += self._previous_measure - current_measure
#         self._previous_measure = current_measure

#         if self._episode_success():
#             reward += self.config.TASK.SUCCESS_REWARD

#         return reward

#     def _episode_success(self):
#         return self._env.get_metrics()[self._success_measure_name]

#     def get_done(self, observations):
#         done = False
#         if self._env.episode_over or self._episode_success():
#             done = True
#         return done

#     def get_info(self, observations):
#         return self.habitat_env.get_metrics()
