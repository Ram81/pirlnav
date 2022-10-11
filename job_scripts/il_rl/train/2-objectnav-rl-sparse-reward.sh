#!/bin/bash
#SBATCH --job-name=onav_rl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=ig-88
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

config="habitat_baselines/config/objectnav/rl/ddppo_rgb_ovrl_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_rl/ddppo_hm3d/rgb_ovrl_with_augs/sparse_reward/hm3d_v0_1_0/seed_1_debug/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_rl/ddppo_hm3d/rgb_ovrl_with_augs/sparse_reward/hm3d_v0_1_0/seed_1_debug/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 40000 \
NUM_PROCESSES 8 \
RL.DDPPO.distrib_backend "NCCL" \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE 0.1 \
MODEL.CRITIC.detach_critic_input False \
