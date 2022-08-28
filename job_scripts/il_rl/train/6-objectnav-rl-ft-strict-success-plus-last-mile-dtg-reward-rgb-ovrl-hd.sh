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

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/rgb_ovrl/strict_success_v2_last_mile_dtg_reward/hm3d_v0_1_0/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/rgb_ovrl/strict_success_v2_last_mile_dtg_reward/hm3d_v0_1_0/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/rgb_ovrl/seed_1/ObjectNav_omnidata_DINO_02_77k.pth"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 20000 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "NCCL" \
RL.Finetune.start_actor_finetuning_at 750 \
RL.Finetune.actor_lr_warmup_update 1500 \
RL.Finetune.start_critic_warmup_at 500 \
RL.Finetune.critic_lr_decay_update 1000 \
TASK_CONFIG.DATASET.SPLIT "train_aug" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE 0.1 \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'TRAIN_SUCCESS', 'STRICT_SUCCESS', 'ANGLE_TO_GOAL', 'ANGLE_SUCCESS', 'SIMPLE_REWARD']" \
TASK_CONFIG.TASK.SIMPLE_REWARD.SLACK_PENALTY 0.0 \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD False \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_STRICT_SUCCESS_REWARD_V2 True \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD True \
MODEL.hm3d_goal False \
MODEL.embed_sge False \
MODEL.USE_SEMANTICS False \
MODEL.USE_PRED_SEMANTICS False \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda False \
MODEL.SEMANTIC_PREDICTOR.name "rednet" \
MODEL.RGB_ENCODER.cnn_type "VisualEncoder" \
MODEL.RGB_ENCODER.backbone "resnet50" \
MODEL.RGB_ENCODER.use_augmentations False \
MODEL.RGB_ENCODER.augmentations_name "+" \
MODEL.RGB_ENCODER.freeze_backbone False \
MODEL.RGB_ENCODER.randomize_augmentations_over_envs False \
MODEL.RGB_ENCODER.pretrained_encoder "data/visual_encoders/omnidata_DINO_02.pth"

