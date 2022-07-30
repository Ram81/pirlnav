export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k"
TENSORBOARD_DIR="tb/objectnav_scene_graph/overfitting/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_scene_graph/overfitting/seed_1/"
INFLECTION_COEF=1.721878842109143
set -x

echo "In ObjectNav IL DDP"
python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 500 \
NUM_UPDATES 5000 \
NUM_PROCESSES 4 \
LOG_INTERVAL 1 \
IL.BehaviorCloning.num_steps 64 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "sample" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.CONTENT_SCENES "['GTV2Y73Sn5t']" \
MODEL.hm3d_goal True \
MODEL.USE_SEMANTICS False \
MODEL.USE_PRED_SEMANTICS False \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda True \
