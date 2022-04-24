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
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

ddp=$1

if [[ $ddp == "ddp" ]]; then
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_ddp_pickplace_mp3d.yaml --run-type eval
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/pickplace/il_pickplace_mp3d.yaml --run-type eval
fi
