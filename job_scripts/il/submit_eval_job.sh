#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/eval/ddpil-%j.out
#SBATCH --error=slurm_logs/eval/ddpil-%j.err
#SBATCH --requeue

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_77k"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred_val/128gpus/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/sem_seg_pred/seed_1/ckpt.11.pth"
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 20 \
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
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
MODEL.embed_sge True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
