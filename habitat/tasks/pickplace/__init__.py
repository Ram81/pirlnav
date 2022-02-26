#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_pickplace_task():
    try:
        from habitat.tasks.pickplace.pickplace import PickPlaceTask  # noqa
    except ImportError as e:
        rearrangement_task_import_error = e

        @registry.register_task(name="PickPlaceTask-v0")
        class PickPlaceTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise rearrangement_task_import_error
