#!/bin/bash
source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate il-rl

cd /srv/flash1/rramrakhya6/spring_2022/habitat-imitation-baselines
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

path=$1
task=$2
output_prefix=$3

if [[ $task == "objectnav" ]]; then
    echo "in ObjectNav replay"
    python examples/objectnav_replay.py --path $path --output-prefix $output_prefix
else
    python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix $prefix --restore-state
fi

rm sample_unapproved_hits.zip

python psiturk_dataset/utils/upload_files_to_s3.py --file demos/ --s3-path data/hit_data/video/$prefix
python psiturk_dataset/utils/upload_files_to_s3.py --file instructions.json --s3-path data/hit_data/instructions.json

rm instructions.json
rm demos/*

# current_dt=$(date '+%Y-%m-%d')
# cp data/hit_data/visualisation/hits.json.gz data/live_hits/live_hits_${current_dt}.json.gz
