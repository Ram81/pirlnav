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

config=$1
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d/sem_seg_pred/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_3/ckpt_22_metrics/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d/sem_seg_pred/sparse_reward/policy_warmup_critic_decay_mlp/train_split/seed_3/ckpt.22.pth"
set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 8 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
EVAL.SPLIT "val" \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
MODEL.embed_sge True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
