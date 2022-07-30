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
conda activate scene-graph

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR "tb/objectnav_il_rl_ft/ddppo/rgbd/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_1/" \
CHECKPOINT_FOLDER "data/new_checkpoints/objectnav_il_rl_ft/ddppo/rgbd/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_1/" \
RL.DDPPO.distrib_backend "GLOO" \
RL.Finetune.start_actor_finetuning_at 1500 \
RL.Finetune.actor_lr_warmup_update 3000 \
RL.Finetune.start_critic_warmup_at 1000 \
RL.Finetune.critic_lr_decay_update 2000 \
