#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Training..."
echo "Hab-Sim: ${PYTHONPATH}"

ddp=$1

if [[ $ddp == "ddp" ]]; then
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_ddp_pickplace_mp3d.yaml --run-type train
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_pickplace_mp3d.yaml --run-type train
fi
