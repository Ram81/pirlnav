#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@1000
#SBATCH --partition=long
#SBATCH --constraint=a40
#SBATCH --exclude=robby
#SBATCH --output=slurm_logs/ddpil-%j.out
#SBATCH --error=slurm_logs/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate scene-graph

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1


DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k"
TENSORBOARD_DIR="tb/objectnav_scene_graph/objectnav_hm3d/objectnav_hm3d_10k/rgbd_spatial_graph/bbox_feats_edgeconv_step_bfix/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_scene_graph/objectnav_hm3d/objectnav_hm3d_10k/rgbd_spatial_graph/bbox_feats_edgeconv_step_bfix/seed_1/"
INFLECTION_COEF=3.234951275740812
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
LOG_INTERVAL 10 \
IL.BehaviorCloning.num_steps 32 \
IL.BehaviorCloning.num_mini_batch 2 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
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
MODEL.DEPTH_ENCODER.cnn_type "VlnResnetDepthEncoder" \
MODEL.DETECTOR.config "configs/detector/mask_rcnn/mask_rcnn_r50_150k_256x256.py" \
MODEL.DETECTOR.checkpoint_path "data/new_checkpoints/mmdet/detector/mask_rcnn_r50_1496cat_150k_ds_256x256.pth"