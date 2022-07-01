#!/bin/bash
#SBATCH --job-name=onav_eval
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --constraint=volta32gb
#SBATCH --output=slurm_logs/eval/eval-%j.out
#SBATCH --error=slurm_logs/eval/eval-%j.err

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
checkpoint=$2

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_77k"

set -x

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

srun python -u -m habitat_baselines.run \
--exp-config  $config \
--run-type eval \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
EVAL_CKPT_PATH_DIR $checkpoint \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']"