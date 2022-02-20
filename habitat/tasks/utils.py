#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import quaternion  # noqa # pylint: disable=unused-import

from habitat.sims.habitat_simulator.actions import HabitatSimActions


def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j ** 2 + q_k ** 2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i ** 2 + q_k ** 2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i ** 2 + q_j ** 2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def compute_pixel_coverage(instance_seg, object_id):
    cand_mask = instance_seg == object_id
    score = cand_mask.sum().astype(np.float64) / cand_mask.size
    return score


def get_habitat_sim_action(action):
    if action == "TURN_RIGHT":
        return HabitatSimActions.TURN_RIGHT
    elif action == "TURN_LEFT":
        return HabitatSimActions.TURN_LEFT
    elif action == "MOVE_FORWARD":
        return HabitatSimActions.MOVE_FORWARD
    elif action == "MOVE_BACKWARD":
        return HabitatSimActions.MOVE_BACKWARD
    elif action == "LOOK_UP":
        return HabitatSimActions.LOOK_UP
    elif action == "LOOK_DOWN":
        return HabitatSimActions.LOOK_DOWN
    return HabitatSimActions.STOP


def get_habitat_sim_action_str(action):
    if action == HabitatSimActions.TURN_RIGHT:
        return "turnRight"
    elif action == HabitatSimActions.TURN_LEFT:
        return "turnLeft"
    elif action == HabitatSimActions.MOVE_FORWARD:
        return "moveForward"
    elif action == HabitatSimActions.MOVE_BACKWARD:
        return "moveBackward"
    elif action == HabitatSimActions.LOOK_UP:
        return "lookUp"
    elif action == HabitatSimActions.LOOK_DOWN:
        return "lookDown"
    elif action == HabitatSimActions.GRAB_RELEASE:
        return "grabReleaseObject"
    elif action == HabitatSimActions.NO_OP:
        return "stepPhysics"
    elif action == HabitatSimActions.START:
        return "start"
    return "stop"
