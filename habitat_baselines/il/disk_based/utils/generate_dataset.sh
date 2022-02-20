#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Generate IL episode dataset"
echo "hab sim: ${PYTHONPATH}"

scene=$1
task=$2
path=$3
semantic=$4

if [[ $semantic == "semantic" ]]; then
    echo "Add semantic obs"
    python habitat_baselines/rearrangement/utils/generate_dataset.py --episodes $path --mode train --scene $scene --task $task --use-semantic
elif [[ $semantic == "novision" ]]; then
    echo "Add no vision obs"
    python habitat_baselines/rearrangement/utils/generate_dataset.py --episodes $path --mode train --scene $scene --task $task --no-vision
else
    python habitat_baselines/rearrangement/utils/generate_dataset.py --episodes $path --mode train --scene $scene --task $task
fi
