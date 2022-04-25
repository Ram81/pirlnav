#!/bin/bash
#SBATCH --job-name=onav_il
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --qos=ram-special
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/eval/ddpil-%j.out
#SBATCH --error=slurm_logs/eval/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type eval \
NUM_PROCESSES 1 \
TENSORBOARD_DIR "tb/objectnav_il/objectnav_mp3d_35k/rgbd/seed_1/ckpt_14/" \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
EVAL.SPLIT "val" \
EVAL_CKPT_PATH_DIR "data/new_checkpoints/objectnav/objectnav_mp3d_35k/rgbd/seed_1/ckpt.14.pth" \
MODEL.hm3d_goal "False" \
MODEL.embed_sge "False" \
MODEL.USE_SEMANTICS "False" \
MODEL.USE_PRED_SEMANTICS "False" \
