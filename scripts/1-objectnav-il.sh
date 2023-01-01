#!/bin/bash
#SBATCH --job-name=pirlnav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddpil-%j.out
#SBATCH --error=slurm_logs/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-web

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="configs/experiments/il_ddp_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/"
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
CHECKPOINT_INTERVAL 100 \
NUM_UPDATES 20000 \
NUM_PROCESSES 8 \
LOG_INTERVAL 10 \
IL.BehaviorCloning.num_steps 64 \
IL.BehaviorCloning.num_mini_batch 2 \
IL.BehaviorCloning.wd 0.0 \
SENSORS "['RGB_SENSOR']" \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
TASK_CONFIG.DATASET.MAX_EPISODE_STEPS 500 \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'DEMONSTRATION_SENSOR', 'INFLECTION_WEIGHT_SENSOR']" \
WANDB_NAME "pirlnav-ovrl-resnet50-train-split-hab-v1" \
