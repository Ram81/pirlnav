#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_pickplace_dataset():
    try:
        from habitat.datasets.pickplace.pickplace_dataset import (  # noqa: F401 isort:skip
            PickPlaceDatasetV1,
        )
    except ImportError as e:
        rearrangement_dataset_import_error = e

        @registry.register_dataset(name="RearrangementDS-v1")
        class PickPlaceDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise rearrangement_dataset_import_error
