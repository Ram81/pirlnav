#!/bin/bash
#SBATCH --job-name=onav_eval
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/eval/eval-%j.out
#SBATCH --error=slurm_logs/eval/eval-%j.err

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate scene-graph

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

path=$1
val_dataset_path=$2
checkpoint=$3

set -x

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

srun python -u -m habitat_baselines.run \
--exp-config  $path \
--run-type eval \
TASK_CONFIG.DATASET.DATA_PATH "$val_dataset_path/{split}/{split}.json.gz" \
EVAL_CKPT_PATH_DIR $checkpoint \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']"