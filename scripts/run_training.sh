#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting training"
echo "hab sim: ${PYTHONPATH}"

python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/il_pickplace_mp3d.yaml --run-type train
