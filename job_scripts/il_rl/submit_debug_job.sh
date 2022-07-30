#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=debug
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddp-il-rl-%j.out
#SBATCH --error=slurm_logs/ddp-il-rl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate scene-graph

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/overfitting/sem_seg_pred/sparse_reward_success_0.5/train_split/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/overfitting/sem_seg_pred/sparse_reward_success_0.5/train_split/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred/seed_1/ckpt.28.pth"
set -x

echo "In ObjectNav IL+RL Debug"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 5000 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "GLOO" \
RL.Finetune.start_actor_finetuning_at 10 \
RL.Finetune.actor_lr_warmup_update 50 \
RL.Finetune.start_critic_warmup_at 10 \
RL.Finetune.critic_lr_decay_update 40 \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
