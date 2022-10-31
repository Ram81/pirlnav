#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint="a40"
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

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_20k/rgb_ovrl/seed_1/hm3d_v0_1_0_evals/ckpt_38_val_videos/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_20k/rgb_ovrl/seed_1/ckpt.38.pth"
VIDEO_DIR="video_dir/objectnav_il/objectnav_hm3d/objectnav_hm3d_20k/rgb_ovrl/seed_1/hm3d_v0_1_0_evals/ckpt_38_val_videos/"
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 20 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
VIDEO_OPTION "['disk']" \
VIDEO_DIR $VIDEO_DIR \
EVAL.SPLIT "val" \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
EVAL.EVAL_FREQ 4 \
EVAL.FIRST_EVAL_INDEX 98 \
EVAL.USE_CKPT_CONFIG False \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL', 'TOP_DOWN_MAP', 'EXPLORATION_METRICS', 'BEHAVIOR_METRICS']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \


# 404140 - uuid