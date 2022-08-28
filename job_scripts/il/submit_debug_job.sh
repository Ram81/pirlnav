#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=debug,user-overcap
#SBATCH --constraint=a40
#SBATCH --exclude=robby
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

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_fe_70k_balanced"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/fronteir_exp/seed_2/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/fronteir_exp/seed_2/"
INFLECTION_COEF=3.234951275740812
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 100 \
NUM_UPDATES 5000 \
NUM_PROCESSES 8 \
LOG_INTERVAL 10 \
IL.BehaviorCloning.num_steps 64 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
MODEL.USE_SEMANTICS True \
MODEL.USE_PRED_SEMANTICS True \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
MODEL.SEMANTIC_PREDICTOR.name "rednet" \
TASK_CONFIG.DATASET.CONTENT_SCENES "['MVVzj944atG', 'TSJmdttd2GV', 'qk9eeNeR4vw', 'u9rPN5cHWBg', 'DoSbsoo4EAg', 'gmuS7Wgsbrx', '6imZUJGRUq4', 'vDfkYo5VqEQ', 'ggNAcMh8JPT', 'QN2dRqwd84J', 'b3WpMbPFB6q', 'YHmAkqgwe2p', 'qz3829g1Lzf', 'URjpCob8MGw', 'RaYrxWt5pR1', 'YJDUB7hWg9h', 'gQ3xxshDiCz', 'VoVGtfYrpuQ', 'oEPjPNSPmzL', 'yHLr6bvWsVm', 'LcAd9dhvVwh', '8wJuSPJ9FXG', 'wsAYBFtQaL7', '5biL7VEkByM', 'xAHnY3QzFUN', 'HxmXPBbFCkH', 'fxbzYAGkrtm', '77mMEyxhs44', '226REUyJh2K', 'GtM3JtRvvvR', 'W9YAR9qcuvN', 'gjhYih4upQ9', 'HfMobPm86Xn', 'ACZZiU6BXLz', 'v7DzfFFEpsD', 'YmWinf3mhb5', 'GGBvSFddQgs', 'xWvSkKiWQpC', 'vLpv2VX547B', '3XYAD64HpDr', 'oahi4u45xMf', 'E1NrAhMoqvB', 'GTV2Y73Sn5t', 'NGyoyh91xXJ', 'nS8T59Aw3sf', 'pcpn6mFqFCg', 'CthA7sQNTPK', '1S7LAXRdDqK', 'JptJPosx1Z6', 'TYDavTf8oyy', 'YMNvYDhK8mB', 'ixTj1aTMup2', 'Jfyvj3xn2aJ', 'Z2DQddYp1fn', 'wPLokgvCnuk', 'g7hUFVNac26', 'iKFn6fzyRqs', 'Wo6kuutE9i7', 'U3oQjwTuMX8', 'hWDDQnSDMXb', 'xgLmjqzoAzF', 'h6nwVLpAKQz', '1UnKg1rAb8A', '3CBBjsNkhqW', 'FRQ75PjD278', 'XiJhRLvpKpX', 'j6fHrce9pHR', 'nACV8wLu1u5', 'QVAA6zecMHu', 'FnDDfrBZPhh', '4vwGX7U38Ux', 'HeSYRw7eMtG', 'NtnvZSMK3en', 'DNWbUAJYsPy']" \

