#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting eval"
echo "hab sim: ${PYTHONPATH}"

distrib=$1

if [[ $distrib == "distrib" ]]; then
    echo "in distrib eval"
    python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/il_distrib_object_rearrangement.yaml --run-type eval
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml --run-type eval
fi
