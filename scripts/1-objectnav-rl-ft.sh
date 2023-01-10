#!/bin/bash
#SBATCH --job-name=pirlnav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddprl-%j.out
#SBATCH --error=slurm_logs/ddprl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pirlnav

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="configs/experiments/rl_ft_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ovrl_resnet50/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/ovrl_resnet50/seed_1/"
PRETRAINED_WEIGHTS=$1

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
srun python -u -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 20000 \
NUM_ENVIRONMENTS 8 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \

