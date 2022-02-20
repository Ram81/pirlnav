#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting ppo eval"
echo "hab sim: ${PYTHONPATH}"

sensor=$1

if [[ $sensor == "pos" ]]; then
    echo "DDPPO pos eval"
    python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/ddppo_object_rearrangement_pos.yaml --run-type eval
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/ddppo_object_rearrangement.yaml --run-type eval
fi
