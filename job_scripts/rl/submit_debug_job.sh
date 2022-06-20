#!/bin/bash
#SBATCH --job-name=onav_rl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=long
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddp-rl-%j.out
#SBATCH --error=slurm_logs/ddp-rl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="habitat_baselines/config/objectnav/rl/ddppo_count_explore_semseg_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_rl/ddppo_hm3d/sem_seg_pred/count_explore_reward/sample_split/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_rl/ddppo_hm3d/sem_seg_pred/count_explore_reward/sample_split/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 50000 \
NUM_PROCESSES 8 \
RL.DDPPO.distrib_backend "GLOO" \
TASK_CONFIG.DATASET.SPLIT "sample" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'COVERAGE']" \
MODEL.hm3d_goal True \
