#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@300
#SBATCH --partition=long,user-overcap
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddp-il-rl-%j.out
#SBATCH --error=slurm_logs/ddp-il-rl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d/sem_seg_pred/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_3/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d/sem_seg_pred/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_3/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/"
PRETRAINED_WEIGHTS="/srv/flash1/rramrakhya6/habitat-web/habitat-lab/data/new_checkpoints/objectnav/objectnav_hm3d_hd_20k_ft/sem_seg_pred/seed_2/ckpt.22.pth"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 50000 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "GLOO" \
RL.Finetune.start_actor_finetuning_at 1500 \
RL.Finetune.actor_lr_warmup_update 3000 \
RL.Finetune.start_critic_warmup_at 1000 \
RL.Finetune.critic_lr_decay_update 2000 \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \