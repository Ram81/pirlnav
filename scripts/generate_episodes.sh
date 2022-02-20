#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

scene=$1
output_path=$2

echo "Starting episode generation!"
echo "data/scene_datasets/habitat-test-scenes/${scene}.glb"


python psiturk_dataset/task/generate_object_locations.py --scenes data/scene_datasets/habitat-test-scenes/$scene.glb --task-config psiturk_dataset/task/rearrangement.yaml --number_retries_per_target 5000 --output $output_path -n 5 --use_google_objects
