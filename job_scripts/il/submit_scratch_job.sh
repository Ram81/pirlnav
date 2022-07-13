#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --time=3-00:00:00
#SBATCH --gres gpu:8
#SBATCH --nodes 16
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@1000
#SBATCH --mem=480GB
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/ddpil-%j.out
#SBATCH --error=slurm_logs/ddpil-%j.err
#SBATCH --requeue

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_77k"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/il_model_train/robot_camera_settings_without_noise_and_coco_detector"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/il_model/robot_camera_settings_without_noise_and_coco_detector"
INFLECTION_COEF=3.186409513220174
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
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.SEMANTIC_PREDICTOR.REDNET.pretrained_weights "data/rednet-models/rednet_semmap_mp3d_40_v2_vince.pth" \
MODEL.SEMANTIC_PREDICTOR.REDNET.num_classes 29 \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
