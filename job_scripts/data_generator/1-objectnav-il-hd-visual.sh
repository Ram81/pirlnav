#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint="a40"
#SBATCH --output=slurm_logs/data/gen-%j.out
#SBATCH --error=slurm_logs/data/gen-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate il-rl

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

set -x

echo "In ObjectNav IL DDP"
srun python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/rgb_ovrl/seed_1/hm3d_v0_1_0_evals/ckpt_best_val_replays/evaluation_meta.json \
--baseline il_hd
