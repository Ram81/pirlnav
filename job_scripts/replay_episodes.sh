#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

prefix=$1
task=$2

if [[ $task == "objectnav" ]]; then
    python psiturk_dataset/parsing/parse_objectnav_dataset.py --replay-path data/hit_data/visualisation/unapproved_hits  --output-path data/datasets/objectnav_gibson_v2/train/content
    python examples/objectnav_replay.py --replay-episode data/datasets/objectnav_gibson_v2/train/train.json.gz --step-env --output-prefix $prefix
else
    python psiturk_dataset/parsing/parser.py --replay-path data/hit_data/visualisation/unapproved_hits --output-path data/hit_data/visualisation/hits.json
    python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix $prefix --restore-state
fi

