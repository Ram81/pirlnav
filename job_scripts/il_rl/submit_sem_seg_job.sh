#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --time=3-00:00:00
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@300
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/ddp-il-rl-%j.out
#SBATCH --error=slurm_logs/ddp-il-rl-%j.err
#SBATCH --requeue

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred_rl_fine_tuning/128gpus/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred_rl_fine_tuning/128gpus/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1_fixed/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred/seed_1/ckpt.11.pth"
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
RL.Finetune.start_actor_finetuning_at 375 \
RL.Finetune.actor_lr_warmup_update 750 \
RL.Finetune.start_critic_warmup_at 250 \
RL.Finetune.critic_lr_decay_update 500 \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.SEMANTIC_PREDICTOR.REDNET.pretrained_weights "data/rednet-models/rednet_semmap_mp3d_40_v2_vince.pth" \
MODEL.SEMANTIC_PREDICTOR.REDNET.num_classes 29 \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
