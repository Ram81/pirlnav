#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 2
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

config="habitat_baselines/config/objectnav/il/il_rgb_ddp_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k/"
TENSORBOARD_DIR="tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_10k/debug_rgb_ovrl/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/objectnav_hm3d/objectnav_hm3d_10k/debug_rgb_ovrl/seed_1/"
INFLECTION_COEF=3.1915100047989653
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 250 \
NUM_UPDATES 20000 \
NUM_PROCESSES 8 \
IL.BehaviorCloning.encoder_lr 1e-4 \
IL.BehaviorCloning.num_steps 32 \
IL.BehaviorCloning.num_mini_batch 2 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "overfitting" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.RGB_ENCODER.backbone "vit_small_patch16" \
MODEL.RGB_ENCODER.pretrained_encoder "data/visual_encoders/mae_vit_small_decoder_large_HGPS_RE10K_100.pth" \
TASK_CONFIG.DATASET.CONTENT_SCENES "['XiJhRLvpKpX', 'xWvSkKiWQpC', 'yHLr6bvWsVm', 'YHmAkqgwe2p', 'YJDUB7hWg9h', 'YMNvYDhK8mB', 'YmWinf3mhb5', 'Z2DQddYp1fn']" \
TASK_CONFIG.DATASET.MAX_EPISODE_STEPS 1000 

