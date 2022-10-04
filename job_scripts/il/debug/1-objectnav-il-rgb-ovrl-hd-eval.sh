#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/eval/ddpil-%j.out
#SBATCH --error=slurm_logs/eval/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="habitat_baselines/config/objectnav/il/il_rgb_ddp_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_10k/debug_rgb_ovrl/seed_1/ckpt_7_val/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_10k/debug_rgb_ovrl/seed_1/ckpt.7.pth"
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 8 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
EVAL.SPLIT "overfitting" \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL', 'TOP_DOWN_MAP', 'EXPLORATION_METRICS']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.RGB_ENCODER.backbone "vit_small_patch16" \
MODEL.RGB_ENCODER.pretrained_encoder "data/visual_encoders/mae_vit_small_decoder_large_HGPS_RE10K_100.pth" \
TASK_CONFIG.DATASET.CONTENT_SCENES "['XiJhRLvpKpX', 'xWvSkKiWQpC', 'yHLr6bvWsVm', 'YHmAkqgwe2p', 'YJDUB7hWg9h', 'YMNvYDhK8mB', 'YmWinf3mhb5', 'Z2DQddYp1fn']" \


# 404140 - uuid