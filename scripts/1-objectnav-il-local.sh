#!/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_il/overfitting/ovrl_resnet50/seed_2/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50/seed_2/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_2k/"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
python -u -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 1000 \
NUM_ENVIRONMENTS 16 \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
