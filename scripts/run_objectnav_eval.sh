#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting eval"
echo "hab sim: ${PYTHONPATH}"

path=$1
python habitat_baselines/run.py --exp-config $path --run-type eval
