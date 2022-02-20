#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

prefix=$1
task=$2

# wget https://habitat-on-web.s3.amazonaws.com/data/hit_data/instructions.json
#wget https://habitat-on-web.s3.amazonaws.com/data/hit_data/unapproved_hits.zip

#unzip -o unapproved_hits.zip 

if [[ $task == "objectnav" ]]; then
    echo "in ObjectNav parsing"
    # Append to dataset
    # python psiturk_dataset/parsing/parse_objectnav_dataset.py --replay-path data/hit_data/visualisation/unapproved_hits  --output-path data/datasets/objectnav_mp3d/objectnav_mp3d_thda_50k/train/content/ --append-dataset
    # Generate replays
    # python psiturk_dataset/parsing/parse_objectnav_dataset.py --replay-path data/hit_data/visualisation/unapproved_hits  --output-path data/datasets/objectnav_gibson_v2/train/content --gibson --append-dataset
    python examples/objectnav_replay.py --replay-episode data/datasets/objectnav_gibson_v2/train/train.json.gz --step-env --output-prefix $prefix
else
    python psiturk_dataset/parsing/parser.py --replay-path data/hit_data/visualisation/unapproved_hits --output-path data/hit_data/visualisation/hits.json
    python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix $prefix --restore-state
fi

rm unapproved_hits.zip

python psiturk_dataset/utils/upload_files_to_s3.py --file demos/ --s3-path data/hit_data/video/$prefix
python psiturk_dataset/utils/upload_files_to_s3.py --file instructions.json --s3-path data/hit_data/instructions.json

rm instructions.json
#rm data/hit_data/visualisation/unapproved_hits/*
rm demos/*

# current_dt=$(date '+%Y-%m-%d')
# cp data/hit_data/visualisation/hits.json.gz data/live_hits/live_hits_${current_dt}.json.gz
