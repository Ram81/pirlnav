#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

task=$1
path=$2
meta_file=$3

if [[ $task == "objectnav" ]]; then
    echo "in ObjectNav stats"
    # python examples/objectnav_replay.py --replay-episode data/datasets/objectnav_mp3d_v2/coverage_sample/coverage_sample.json.gz --step-env
    python examples/objectnav_replay.py --replay-episode $path --step-env --output-prefix demos --meta-file $meta_file
else
    python examples/rearrangement_replay.py --replay-episode $path
fi
