# Scene Graphs for EAI

Code for training imitation learning agents for [Objectnav](https://arxiv.org/abs/2006.13171) and [Pick-and-Place](https://arxiv.org/abs/2204.03514) in [Habitat](https://aihabitat.org/).

## Overview

The primary code contributions from the paper are located in:
- Imitation Learning Baselines:
    - ObjectNav: `habitat_baselines/il/env_based/`
    - Pick-and-Place: `habitat_baselines/il/disk_based/`

- Experiment Configurations:
    - ObjectNav: `habitat_baselines/config/objectnav/*.yaml`
    - Pick-and-Place: `habitat_baselines/config/pickplace/*.yaml`

- Replay Scripts:
    - ObjectNav: `examples/objectnav_replay.py`
    - Pick-and-Place: `examples/pickplace_replay.py`

## Installation

1. Install our custom mmdetection repository

    ```bash
    pip install -U openmim
    mim install mmcv-full

    cd mmdetection
    pip install -v -e .
    # "-v" means verbose, or more output
    # "-e" means installing a project in editable mode,
    # thus any local modifications made to the code will take effect without reinstallation.
    ```

1. Clone the repository and install `il_rl_baselines` using the commands below. Note that `python=3.6` is required for working with `il_rl_baselines`. All the development was done on `habitat-lab=0.1.6`.

    ```bash
    cd il_rl_baselines

    # We require python>=3.6 and cmake>=3.10
    conda create -n habitat-web python=3.6 cmake=3.14.0
    conda activate scene-graph

    pip install -e .
    python setup.py develop --all
    ```

1. Install our custom build of `habitat-sim`, we highly recommend using the `habitat-sim` build from source for working with `il_rl_baselines`. Use the following commands to set it up:

    ```bash
    git clone git@github.com:Ram81/habitat-sim.git
    cd habitat-sim
    ```

1. Install dependencies

    Common

   ```bash
   pip install -r requirements.txt
   ```

    Linux (Tested with Ubuntu 18.04 with gcc 7.4.0)

   ```bash
   sudo apt-get update || true
   # These are fairly ubiquitous packages and your system likely has them already,
   # but if not, let's get the essentials for EGL support:
   sudo apt-get install -y --no-install-recommends \
        libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
   ```

   See this [configuration for a full list of dependencies](https://github.com/facebookresearch/habitat-sim/blob/master/.circleci/config.yml#L64) that our CI installs on a clean Ubuntu VM. If you run into build errors later, this is a good place to check if all dependencies are installed.

1. Build Habitat-Sim

    Default build with bullet (for machines with a display attached)

   ```bash
   # Assuming we're still within habitat conda environment
   ./build.sh --bullet
   ```

    For headless systems (i.e. without an attached display, e.g. in a cluster) and multiple GPU systems

   ```bash
   ./build.sh --headless --bullet
   ```

1. For use with il_rl_baselines and your own python code, add habitat-sim to your `PYTHONPATH`. For example modify your `.bashrc` (or `.bash_profile` in Mac OS X) file by adding the line:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/habitat-sim/
   ```


## Data

- Symlink data from `/data/datasets/tmp/ramrakhya_share/data/` in `data/ using

  ```bash
  ln -s /data/datasets/tmp/ramrakhya_share/data/ data
  ```

### Dataset Folder Structure

The code requires the datasets in `data` folder in the following format:

  ```bash
  ├── il_rl_baselines/
  │  ├── data
  │  │  ├── scene_datasets/
  │  │  │  ├── mp3d/
  │  │  │  │  ├── JeFG25nYj2p.glb
  │  │  │  │  └── JeFG25nYj2p.navmesh
  │  │  ├── datasets
  │  │  │  ├── objectnav/
  │  │  │  │  ├── objectnav_mp3d_70k/
  │  │  │  │  │  ├── train/
  │  │  │  │  │  ├── sample/
  ```


### Test Setup

To verify that the data is set up correctly, run:

  ```bash
  python examples/objectnav_replay.py --path data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_70k/sample/sample.json.gz  --detector-config configs/detector/mask_rcnn/mask_rcnn_r50_150k_256x256.py --detector-checkpoint data/new_checkpoints/mmdet/detectormask_rcnn_r50_1496cat_150k_ds_256x256.pth  --save-step-image --save-videos --num-episodes 1
  ```


## Usage

### Training


For training the behavior cloning policy on the ObjectGoal Navigation task using the environment based setup:
    
1. Use the following script for multi-node training

  ```bash
  sbatch job_scripts/il/submit_scratch_job.sh habitat_baselines/config/objectnav/il_ddp_objectnav.yaml
  ```

### Evaluation


For evaluating a checkpoint on the ObjectGoal Navigation task using the environment based setup:
    
1. Use the following script if trained using distributed setup

  ```bash
  sbatch job_scripts/il/submit_eval_job.sh habitat_baselines/config/objectnav/il_ddp_objectnav.yaml
  ```
