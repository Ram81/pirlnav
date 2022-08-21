#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:8
#SBATCH --nodes 2
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@300
#SBATCH --partition=long
#SBATCH --constraint=a40
#SBATCH --exclude=robby
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

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/sem_seg_pred/sparse_success_atg_reward_ckpt_28/hm3d_v0_1_0/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/sem_seg_pred/sparse_success_atg_reward_ckpt_28/hm3d_v0_1_0/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred/seed_1/ckpt.28.pth"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 40000 \
ENV_NAME "ObjectNavDenseRewardEnv" \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "NCCL" \
RL.Finetune.start_actor_finetuning_at 750 \
RL.Finetune.actor_lr_warmup_update 1500 \
RL.Finetune.start_critic_warmup_at 500 \
RL.Finetune.critic_lr_decay_update 1000 \
RL.REWARD_MEASURE "simple_reward" \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE 0.1 \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'TRAIN_SUCCESS', 'STRICT_SUCCESS', 'ANGLE_TO_GOAL', 'ANGLE_SUCCESS', 'SIMPLE_REWARD']" \
TASK_CONFIG.TASK.SIMPLE_REWARD.SLACK_PENALTY 0.0 \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD False \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD_V2 False \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD False \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_SLACK_PENALTY False \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ANGLE_SUCCESS_REWARD True \
MODEL.hm3d_goal True \
MODEL.embed_sge True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
MODEL.SEMANTIC_PREDICTOR.name "rednet" \
