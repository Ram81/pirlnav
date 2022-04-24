#!/bin/bash
#SBATCH --job-name=ddp_onav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/ddppo-%j.out
#SBATCH --error=slurm_logs/ddppo-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

ddp=$1

if [[ $ddp == "ddp" ]]; then
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_ddp_pickplace_mp3d.yaml --run-type train
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_pickplace_mp3d.yaml --run-type train
fi
