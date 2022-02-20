#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting parser"
echo "hab sim: ${PYTHONPATH}"

task=$1

if [[ $task == "objectnav" ]]; then
    python psiturk_dataset/parsing/parser.py --replay-path data/hit_data/visualisation/unapproved_hits/ --output-path data/hit_approvals/hits_max_length_1500.json --max-episode-length 1500
else
    python psiturk_dataset/generator/objectnav_shortest_path_generator.py --episodes data/datasets/objectnav_mp3d_v4/train/train.json.gz
fi
