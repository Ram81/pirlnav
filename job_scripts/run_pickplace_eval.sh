#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

ddp=$1

if [[ $ddp == "ddp" ]]; then
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_ddp_pickplace_mp3d.yaml --run-type eval
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_pickplace_mp3d.yaml --run-type eval
fi
