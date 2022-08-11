#!/bin/bash
#SBATCH --job-name=onav_ilrl
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=robby
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

config="habitat_baselines/config/objectnav/il_rl/ddppo_semseg_ft_objectnav_ensemble.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/sem_seg_pred/sparse_reward/train_split/seed_1_4node/ensemble/v0_evals/ckpt_4_pt_32_w5_5_val/"
EVAL_CKPT_PATH_DIR_POLICY_A="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/sem_seg_pred/sparse_reward/train_split/seed_1_4node/ckpt.32.pth"
EVAL_CKPT_PATH_DIR_POLICY_B="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/sem_seg_pred/sparse_reward_pt_ckpt_4/train_split/seed_1/ckpt.16.pth"
# EVAL_CKPT_PATH_DIR_POLICY_B="data/new_checkpoints/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/sem_seg_pred/sparse_reward/train_split/seed_1_4node/ckpt.32.pth"

set -x

echo "In ObjectNav IL+RL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 20 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
EVAL.ENSEMBLE True \
EVAL.SPLIT "val" \
EVAL.USE_CKPT_CONFIG False \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL_CKPT_PATH_DIR_POLICY_A $EVAL_CKPT_PATH_DIR_POLICY_A \
EVAL_CKPT_PATH_DIR_POLICY_B $EVAL_CKPT_PATH_DIR_POLICY_B \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL', 'TOP_DOWN_MAP', 'EXPLORATION_METRICS']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
MODEL.embed_sge True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
