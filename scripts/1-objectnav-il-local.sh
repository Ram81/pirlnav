#!/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_il/overfitting/ovrl_resnet50/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50/seed_1/"
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
NUM_ENVIRONMENTS 8 \
RL.DDPPO.force_distributed True \
SENSORS "['RGB_SENSOR']" \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
TASK_CONFIG.DATASET.MAX_EPISODE_STEPS 500 \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'DEMONSTRATION_SENSOR', 'INFLECTION_WEIGHT_SENSOR']" \
