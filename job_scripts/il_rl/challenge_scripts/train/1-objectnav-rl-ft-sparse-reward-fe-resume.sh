#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:8
#SBATCH --nodes 2
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@300
#SBATCH --partition=long
#SBATCH --constraint=a40
#SBATCH --exclude=robby,xaea-12
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

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_fe_70k/sem_seg_pred/sparse_reward_ckpt_3/hm3d_v0_1_0/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_fe_70k/sem_seg_pred/sparse_reward_ckpt_3/hm3d_v0_1_0/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_fe_70k_balanced/sem_seg_pred/seed_1/ckpt.3.pth"
RESUME_STATE_FILE="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_fe_70k/sem_seg_pred/sparse_reward_ckpt_3/hm3d_v0_1_0/seed_1/418861.pth"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 40000 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "NCCL" \
RL.Finetune.start_actor_finetuning_at 750 \
RL.Finetune.actor_lr_warmup_update 1500 \
RL.Finetune.start_critic_warmup_at 500 \
RL.Finetune.critic_lr_decay_update 1000 \
RESUME_STATE_FILE $RESUME_STATE_FILE \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE 0.1 \
MODEL.hm3d_goal True \
MODEL.embed_sge True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
MODEL.SEMANTIC_PREDICTOR.name "rednet" \
