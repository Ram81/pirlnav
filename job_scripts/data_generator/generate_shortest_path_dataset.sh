#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
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

# CONTENT_SCENES="1S7LAXRdDqK,8wJuSPJ9FXG,FRQ75PjD278,GtM3JtRvvvR,ixTj1aTMup2,nS8T59Aw3sf,qz3829g1Lzf,vLpv2VX547B,XiJhRLvpKpX" # split 1
CONTENT_SCENES="1UnKg1rAb8A,ACZZiU6BXLz,fxbzYAGkrtm,GTV2Y73Sn5t,j6fHrce9pHR,NtnvZSMK3en,RaYrxWt5pR1,VoVGtfYrpuQ,xWvSkKiWQpC" # split 2
# CONTENT_SCENES="226REUyJh2K,b3WpMbPFB6q,g7hUFVNac26,h6nwVLpAKQz,Jfyvj3xn2aJ,oahi4u45xMf,TSJmdttd2GV,W16Bm4ysK8v,yHLr6bvWsVm" # split 3
# CONTENT_SCENES="3CBBjsNkhqW,CQWES1bawee,g8Xrdbe9fir,HeSYRw7eMtG,JptJPosx1Z6,oEPjPNSPmzL,TYDavTf8oyy,W9YAR9qcuvN,YHmAkqgwe2p" # split 4
# CONTENT_SCENES="3XYAD64HpDr,CthA7sQNTPK,GGBvSFddQgs,HfMobPm86Xn,LcAd9dhvVwh,pcpn6mFqFCg,U3oQjwTuMX8,Wo6kuutE9i7,YJDUB7hWg9h" # split 5
# CONTENT_SCENES="4vwGX7U38Ux,DNWbUAJYsPy,ggNAcMh8JPT,hWDDQnSDMXb,MVVzj944atG,PPTLa8SkUfo,u9rPN5cHWBg,wPLokgvCnuk,YMNvYDhK8mB" # split 6
# CONTENT_SCENES="5biL7VEkByM,DoSbsoo4EAg,gjhYih4upQ9,HxmXPBbFCkH,nACV8wLu1u5,qk9eeNeR4vw,URjpCob8MGw,wsAYBFtQaL7,YmWinf3mhb5" # split 7
# CONTENT_SCENES="6imZUJGRUq4,E1NrAhMoqvB,gmuS7Wgsbrx,iigzG1rtanx,NEVASPhcrxR,QN2dRqwd84J,v7DzfFFEpsD,xAHnY3QzFUN,Z2DQddYp1fn" # split 8
# CONTENT_SCENES="77mMEyxhs44,FnDDfrBZPhh,gQ3xxshDiCz,iKFn6fzyRqs,NGyoyh91xXJ,QVAA6zecMHu,vDfkYo5VqEQ,xgLmjqzoAzF" # split 9

NUM_EPISODES=1000
INPUT_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/s_path_exclude/content/"
OUTPUT_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_s_path/train_v2/content/"

set -x

echo "In ObjectNav IL DDP"
srun python -u -m examples.objectnav_shortest_path_generator \
--input-path $INPUT_PATH \
--output-path $OUTPUT_PATH \
--num-episodes $NUM_EPISODES \
--scenes $CONTENT_SCENES

