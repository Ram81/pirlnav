#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=long
#SBATCH --qos=ram-special
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/ddpil-%j.out
#SBATCH --error=slurm_logs/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_35k"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/seed_1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav/objectnav_mp3d_thda_70k/sem_seg_pred/seed_1/ckpt.21.pth"
INFLECTION_COEF=3.1902439975324186
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 500 \
NUM_UPDATES 20000 \
NUM_PROCESSES 8 \
IL.BehaviorCloning.lr 0.0001 \
IL.BehaviorCloning.num_steps 64 \
IL.BehaviorCloning.pretrained True \
IL.BehaviorCloning.pretrained_weights $PRETRAINED_WEIGHTS \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
