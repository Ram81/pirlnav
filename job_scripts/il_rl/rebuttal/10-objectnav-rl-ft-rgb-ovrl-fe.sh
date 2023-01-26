#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@300
#SBATCH --partition=long
#SBATCH --constraint=a40
#SBATCH --exclude=dave,ig-88
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

config="habitat_baselines/config/objectnav/il_rl/ddppo_rgb_ovrl_ft_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_fe_70k/rgb_ovrl_with_augs/sparse_reward_ckpt_34/hm3d_v0_1_0/seed_2/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_fe_70k/rgb_ovrl_with_augs/sparse_reward_ckpt_34/hm3d_v0_1_0/seed_2/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_fe_70k_balanced/rgb_ovrl/seed_1/ckpt.34.pth"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
SENSORS "['RGB_SENSOR']" \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 40000 \
NUM_PROCESSES 16 \
RL.PPO.num_mini_batch 4 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "NCCL" \
RL.Finetune.finetune True \
RL.Finetune.start_actor_finetuning_at 750 \
RL.Finetune.actor_lr_warmup_update 1500 \
RL.Finetune.start_critic_warmup_at 500 \
RL.Finetune.critic_lr_decay_update 1000 \
RL.PPO.use_linear_lr_decay True \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.SEED 105