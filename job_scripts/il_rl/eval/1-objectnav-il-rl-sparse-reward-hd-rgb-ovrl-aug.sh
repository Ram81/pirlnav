#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=robby,chappie
#SBATCH --output=slurm_logs/eval/ddp-il-rl-%j.out
#SBATCH --error=slurm_logs/eval/ddp-il-rl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/rgb_ovrl_augs/sparse_reward/hm3d_v0_1_0/seed_1/v0_1_0_evals/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/rgb_ovrl_augs/sparse_reward/hm3d_v0_1_0/seed_1/ckpt.38.pth"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/rgb_ovrl/seed_1/ObjectNav_omnidata_DINO_02_77k_with_augs.pth"

set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
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
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL', 'TOP_DOWN_MAP', 'EXPLORATION_METRICS']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal False \
MODEL.embed_sge False \
MODEL.USE_SEMANTICS False \
MODEL.USE_PRED_SEMANTICS False \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda False \
MODEL.SEMANTIC_PREDICTOR.name "rednet" \
MODEL.RGB_ENCODER.cnn_type "VisualEncoder" \
MODEL.RGB_ENCODER.backbone "resnet50" \
MODEL.RGB_ENCODER.freeze_backbone False \
MODEL.RGB_ENCODER.randomize_augmentations_over_envs False \
MODEL.RGB_ENCODER.pretrained_encoder "data/visual_encoders/omnidata_DINO_02.pth"

# 404148 - uuid
