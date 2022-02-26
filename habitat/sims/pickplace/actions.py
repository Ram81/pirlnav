
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Any

import attr

import habitat_sim
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.core.utils import Singleton

from habitat.core.embodied_task import SimulatorTaskAction
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration
)


@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0


@registry.register_action_space_configuration(name="PickPlaceActions-v0")
class PickPlaceSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def __init__(self, config):
        super().__init__(config)
        # self.config = config
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")
        if not HabitatSimActions.has_action("MOVE_BACKWARD"):
            HabitatSimActions.extend_action_space("MOVE_BACKWARD")
        if not HabitatSimActions.has_action("NO_OP"):
            HabitatSimActions.extend_action_space("NO_OP")
        if not HabitatSimActions.has_action("STOP"):
            HabitatSimActions.extend_action_space("STOP")

    def get(self):
        #config = super().get()
        new_config = {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.MOVE_BACKWARD: habitat_sim.ActionSpec(
                "move_backward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            HabitatSimActions.NO_OP: habitat_sim.ActionSpec(
                "no_op",
                habitat_sim.ActuationSpec(amount=0.05),
            ),
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            ),
        }

        return new_config


@registry.register_task_action
class NoOpAction(SimulatorTaskAction):
    name: str = "NO_OP"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.NO_OP,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.NO_OP)


@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.GRAB_RELEASE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)


@registry.register_task_action
class MoveBackwardAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys():
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD)

