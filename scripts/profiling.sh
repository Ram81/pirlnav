#!/bin/bash

py-spy record --idle --function --subprocesses --rate 50 \
--output pyspy_profile.speedscope --format speedscope -- python psiturk_dataset/generate_object_locations.py --scenes data/scene_datasets/habitat-test-scenes/house_3.glb --task-config psiturk_dataset/rearrangement.yaml --number_retries_per_target 1000 --output task_10_v2.json -n 5