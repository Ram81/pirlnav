#!/bin/bash
#SBATCH --job-name=pirlnav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddprl-%j.out
#SBATCH --error=slurm_logs/ddprl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-web

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="configs/experiments/rl_ft_ddp_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/ckpt.0.pth"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
srun python -u -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 1000 \
NUM_UPDATES 40000 \
NUM_PROCESSES 16 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "NCCL" \
RL.PPO.num_mini_batch 4 \
RL.Finetune.finetune True \
RL.Finetune.start_actor_finetuning_at 50 \
RL.Finetune.actor_lr_warmup_update 50 \
RL.Finetune.start_critic_warmup_at 50 \
RL.Finetune.critic_lr_decay_update 50 \
RL.PPO.use_linear_lr_decay True \
SENSORS "['RGB_SENSOR']" \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
TASK_CONFIG.DATASET.MAX_EPISODE_STEPS 500 \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR']" \
WANDB_NAME "pirlnav-rl-ft" \

