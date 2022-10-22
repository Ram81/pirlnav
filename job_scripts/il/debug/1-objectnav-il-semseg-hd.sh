#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=ig-88
#SBATCH --output=slurm_logs/ddpil-%j.out
#SBATCH --error=slurm_logs/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config="habitat_baselines/config/objectnav/il_ddp_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_medium/"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_medium/debug_semseg_dsize/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_medium/debug_semseg_dsize/seed_1/"
INFLECTION_COEF=3.1915100047989653
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 500 \
NUM_UPDATES 20000 \
NUM_PROCESSES 8 \
IL.BehaviorCloning.num_steps 64 \
IL.BehaviorCloning.num_mini_batch 2 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
MODEL.hm3d_goal True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
MODEL.SEMANTIC_PREDICTOR.name "rednet" \

