#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo "Starting episode generation!"
echo "Episodes v6"

task=$1
sceneId=$2
path=$3
output_path=$4

if [[ $task == "objectnav" ]]; then
    echo "ObjectNav generator"
    python psiturk_dataset/generator/objectnav_shortest_path_generator.py --episodes $path --scene $sceneId --split $sceneId --output-path $output_path
else
    python psiturk_dataset/generator/shortest_path_trajectories.py --output-path $output_path --scene $sceneId --episodes $path
fi
