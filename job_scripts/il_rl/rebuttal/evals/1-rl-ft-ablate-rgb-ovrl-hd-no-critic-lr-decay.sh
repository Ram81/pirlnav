#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=sonny
#SBATCH --output=slurm_logs/eval/ddp-il-rl-%j.out
#SBATCH --error=slurm_logs/eval/ddp-il-rl-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="habitat_baselines/config/objectnav/il_rl/ddppo_rgb_ovrl_ft_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_20k/rgb_ovrl_with_augs/sparse_reward_ckpt_38_no_lr_decay/hm3d_v0_1_0/seed_1/hm3d_v0_1_0_evals/ckpt_20_78_val_seed_2/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_20k/rgb_ovrl_with_augs/sparse_reward_ckpt_38_no_lr_decay/hm3d_v0_1_0/seed_1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_20k/rgb_ovrl/seed_1/ckpt.38.pth"

set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 20 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
EVAL.SPLIT "val" \
EVAL.USE_CKPT_CONFIG False \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL.EVAL_FREQ 4 \
EVAL.FIRST_EVAL_INDEX 62 \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.SEED 104

