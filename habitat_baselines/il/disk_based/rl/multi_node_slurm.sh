#!/bin/bash
#SBATCH --job-name=ddppo_pp
#SBATCH --gres gpu:8
#SBATCH --nodes 2
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 8
#SBATCH --partition=user-overcap
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddppo-%j.out
#SBATCH --error=slurm_logs/ddppo-%j.err

source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

sensor=$1
set -x
if [[ $sensor == "pos" ]]; then
    echo "in pos ppo"
    srun python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/object_rearrangement/ddppo_object_rearrangement_pos.yaml \
    --run-type train
else
    srun python -u -m habitat_baselines.run \
        --exp-config habitat_baselines/config/object_rearrangement/ddppo_object_rearrangement.yaml \
        --run-type train
fi
