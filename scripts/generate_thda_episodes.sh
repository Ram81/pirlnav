#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

python psiturk_dataset/generator/objectnav_thda_generator.py --output-path data/datasets/objectnav_mp3d_thda/train/min_20m --split train --config configs/tasks/objectnav_mp3d_il.yaml --num-episodes 16000