#!/bin/bash
config="habitat_baselines/config/objectnav/il_rl/ddppo_rgb_ovrl_ft_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_20k/rgb_ovrl_with_augs/sparse_reward_ckpt_38/hm3d_v0_1_0/seed_1/hm3d_v0_1_0_evals/ckpt_30_val/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_20k/rgb_ovrl_with_augs/sparse_reward_ckpt_38/hm3d_v0_1_0/seed_1/ckpt.30.pth"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_50k/rgb_ovrl/seed_1/ckpt.110.pth"

set -x

echo "In ObjectNav IL+RL DDP"
python -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 20 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
EVAL.SPLIT "val" \
EVAL.USE_CKPT_CONFIG False \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL', 'TOP_DOWN_MAP', 'EXPLORATION_METRICS', 'BEHAVIOR_METRICS']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \

# 404148 - uuid
