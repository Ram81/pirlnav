#!/bin/bash
source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate scene-graph

cd /srv/flash1/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

prefix=$1
task=$2

if [[ $task == "objectnav" ]]; then
    python psiturk_dataset/parsing/parse_objectnav_dataset.py --path sample_unapproved_hits.zip --output-path $output_path/content/ --scene-dataset $scene_dataset
    python examples/objectnav_replay.py --path data/datasets/objectnav_hm3d_demos/train/train.json.gz --output-prefix $prefix
else
    python psiturk_dataset/parsing/parser.py --replay-path data/hit_data/visualisation/unapproved_hits --output-path data/hit_data/visualisation/hits.json
    python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix $prefix --restore-state
fi

