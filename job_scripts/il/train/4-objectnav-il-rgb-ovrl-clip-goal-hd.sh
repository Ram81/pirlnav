#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=nestor
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

config="habitat_baselines/config/objectnav/il/il_rgb_ddp_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_20k/"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_20k/rgb_ovrl_clip_goal/seed_3/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_20k/rgb_ovrl_clip_goal/seed_3/"
INFLECTION_COEF=3.2080078220377883
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 250 \
NUM_UPDATES 20000 \
NUM_PROCESSES 20 \
IL.BehaviorCloning.num_steps 32 \
IL.BehaviorCloning.num_mini_batch 2 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR', 'DEMONSTRATION_SENSOR', 'INFLECTION_WEIGHT_SENSOR', 'OBJECTGOAL_PROMPT_SENSOR']" \
MODEL.GOAL.use_clip_goal True
