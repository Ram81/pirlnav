#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting objectnav training"
echo "hab sim: ${PYTHONPATH}"

path=$1
python habitat_baselines/run.py --exp-config $path --run-type train
