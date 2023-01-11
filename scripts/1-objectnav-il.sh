#!/bin/bash
#SBATCH --job-name=pirlnav
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddpil-train-%j.out
#SBATCH --error=slurm_logs/ddpil-train-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pirlnav

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

cd /srv/flash1/rramrakhya6/spring_2022/pirlnav

dataset=$1

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/${dataset}"
# TENSORBOARD_DIR="tb/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"
# CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k/"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/ovrl_resnet50/seed_3_wd_zero/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50/seed_3_wd_zero/"
INFLECTION_COEF=3.234951275740812

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
NUM_ENVIRONMENTS 16 \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
