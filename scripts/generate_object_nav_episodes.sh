#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo "Starting ObjectNav dataset generation!"
python examples/generate_objectnav_task_mp3d_all_objs.py train