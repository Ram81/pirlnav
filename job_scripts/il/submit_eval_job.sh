#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=robby
#SBATCH --output=slurm_logs/eval/ddpil-%j.out
#SBATCH --error=slurm_logs/eval/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate scene-graph
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_scene_graph/objectnav_hm3d/objectnav_hm3d_10k/rgbd_spatial_graph/bbox_feats_edgeconv_step_bfix/seed_1/ckpt_28/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_scene_graph/objectnav_hm3d/objectnav_hm3d_10k/rgbd_spatial_graph/bbox_feats_edgeconv_step_bfix/seed_1/ckpt.28.pth"
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 1 \
TENSORBOARD_DIR $TENSORBOARD_DIR \
TEST_EPISODE_COUNT -1 \
EVAL.SPLIT "val" \
EVAL.meta_file "$TENSORBOARD_DIR/evaluation_meta.json" \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
EVAL.USE_CKPT_CONFIG False \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'GOAL_OBJECT_VISIBLE', 'MIN_DISTANCE_TO_GOAL']" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v1" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
MODEL.hm3d_goal True \
MODEL.USE_DETECTOR True \
MODEL.SPATIAL_ENCODER.gcn_type "local_gcn_encoder" \
MODEL.SPATIAL_ENCODER.no_node_cat True \
MODEL.SPATIAL_ENCODER.no_bbox_feats False \
MODEL.SPATIAL_ENCODER.filter_nodes False \
MODEL.SPATIAL_ENCODER.conv_layer "GCNConv" \
MODEL.SPATIAL_ENCODER.out_features_dim 512 \
MODEL.SPATIAL_ENCODER.no_gcn False \
MODEL.SPATIAL_ENCODER.ablate_gcn False \
MODEL.USE_SEMANTICS False \
MODEL.USE_PRED_SEMANTICS False \
MODEL.SEMANTIC_ENCODER.is_hm3d False \
MODEL.SEMANTIC_ENCODER.is_thda False \
MODEL.SEQ2SEQ.use_prev_action True \
MODEL.RGB_ENCODER.cnn_type "ResnetRGBEncoder" \
MODEL.DEPTH_ENCODER.cnn_type "VlnResnetDepthEncoder"

