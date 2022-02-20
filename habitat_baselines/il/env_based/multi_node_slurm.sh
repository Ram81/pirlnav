#!/bin/bash
#SBATCH --job-name=ddp_onav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --qos=ram-special
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/ddppo-%j.out
#SBATCH --error=slurm_logs/ddppo-%j.err
#SBATCH --requeue

#source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

sensor=$1
config=$2
set -x

if [[ $sensor == "env" ]]; then
    echo "In ObjectNav Env DDP"
    srun python -u -m habitat_baselines.run \
    --exp-config $config \
    --run-type train
elif [[ $sensor == "env_single" ]]; then
    echo "In ObjectNav Env DDP"
    srun python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/objectnav/il_ddp_env_single_resnet.yaml \
    --run-type train
elif [[ $sensor == "recollect" ]]; then
    echo "In ObjectNav Recollect DDP"
    srun python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/objectnav/il_recollect_ddp_objectnav.yaml \
    --run-type train
elif [[ $sensor == "il" ]]; then
    echo "In ObjectNav IL DDP"
    srun python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/objectnav/il_objectnav.yaml \
    --run-type train
else
    echo "In ObjectNav IL+RL FT DDP"
    srun python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/objectnav/ddppo_ft_objectnav.yaml \
    --run-type train
fi
