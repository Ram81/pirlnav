#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --qos=ram-special
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/eval/eval-il-rl-%j.out
#SBATCH --error=slurm_logs/eval/eval-il-rl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
TENSORBOARD_DIR "tb/objectnav_il_rl_ft/ddppo/rgbd/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_1/ckpt_39/" \
EVAL_CKPT_PATH_DIR "data/new_checkpoints/objectnav_il_rl_ft/ddppo/rgbd/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_1/ckpt.39.pth"


