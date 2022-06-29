#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=debug,user-overcap
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
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

config=$1

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_s_path"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/seed_1/"
INFLECTION_COEF=1.721878842109143
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 500 \
NUM_UPDATES 5000 \
NUM_PROCESSES 4 \
IL.BehaviorCloning.num_steps 64 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION "[0, 1.31, 0]" \
TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION "[0, 1.31, 0]" \
TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.POSITION "[0, 1.31, 0]" \
TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH 480 \
TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 480 \
TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 480 \
TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT 640 \
TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 640 \
TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 640 \
MODEL.hm3d_goal True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.DEPTH_ENCODER.ddppo_checkpoint "NONE" \
MODEL.SEMANTIC_PREDICTOR.REDNET.pretrained_weights "data/rednet-models/rednet_semmap_mp3d_40_v2_vince.pth" \
MODEL.SEMANTIC_PREDICTOR.REDNET.num_classes 29 \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \