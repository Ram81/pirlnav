# Habitat-Web

Code for training imitation learning agents for [Objectnav](https://arxiv.org/abs/2006.13171) and [PickPlace]() in [Habitat](https://aihabitat.org/). This repo is the official code repository for the paper **[Habitat-Web: Learning Embodied Object-Search from Human Demonstrations at Scale]()**

## Reproducing Results

We provide te best checkpoints for agents trained on ObjectNav and PickPlace. You can use the following checkpoints to reproduce results reported in our paper.

| Task | Split | Checkpoint | Success Rate | SPL |
| --- | --- | --- | --- | --- |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | v1 | [objectnav_semseg.ckpt]() | 27.8 | 9.9 |
| 🆕[PickPlace]() | New Initializations | [pick_place_rgbd_new_inits.ckpt]() | 17.5 | 9.8 |
| 🆕[PickPlace]() | New Instructions | [pick_place_rgbd_new_insts.ckpt]() | 15.1 | 8.3 |
| 🆕[PickPlace]() | New Environments | [pick_place_rgbd_new_envs.ckpt]() | 8.3 | 4.1 |


You can find the pretrained RedNet semantic segmentation model weights [here]().

## Overview

The primary code contributions from the paper are located in:
- Imitation Learning Baselines:
    - ObjectNav: `habitat_baselines/il/env_based/`
    - PickPlace: `habitat_baselines/il/disk_based/`

- Experiment Configurations:
    - ObjectNav: `habitat_baselines/config/objectnav/*.yaml`
    - PickPlace: `habitat_baselines/config/pickplace/*.yaml`

- Replay Scripts:
    - ObjectNav: `examples/objectnav_replay.py`
    - PickPlace: `examples/pickplace_replay.py`

## Installation

1. Clone the repository and install `habitat-web-baselines` using the commands below. Note that `python=3.6` is required for working with `habitat-web-baselines`. All the development was done on `habitat-lab=0.1.6`.

    ```bash
    git clone https://github.com/Ram81/habitat-web-baselines.git
    cd habitat-web-baselines

    # We require python>=3.6 and cmake>=3.10
    conda create -n habitat-web python=3.6 cmake=3.14.0
    conda activate habitat-web

    pip install -e .
    ```

1. Install our custom build of `habitat-sim`, we highly recommend using the `habitat-sim` build from source for working with `habitat-web-baselines`. Use the following commands to set it up:

    ```bash
    git clone --branch habitat-on-web git@github.com:Ram81/habitat-sim.git
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

1. For use with [habitat-web-baselines](https://github.com/Ram81/habitat-web-baselines) and your own python code, add habitat-sim to your `PYTHONPATH`. For example modify your `.bashrc` (or `.bash_profile` in Mac OS X) file by adding the line:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/habitat-sim/
   ```


## Data

### Downloading MP3D scene dataset

- Download the MP3D dataset using the instructions here: https://github.com/facebookresearch/habitat-lab#scenes-datasets (download the full MP3D dataset for use with habitat)

- Move the MP3D scene dataset or create a symlink at data/scene_datasets/mp3d.

### Downloading human demonstrations dataset

- Download the ObjectNav dataset:

    ```bash
    wget https://habitat-on-web.s3.amazonaws.com/release/datasets/objectnav/objectnav_mp3d_70k.zip
    ```

    Unzip the dataset into `data/datasets/objectnav/`


- Download the PickPlace dataset:

    ```bash
    wget https://habitat-on-web.s3.amazonaws.com/release/datasets/pick_place/pick_place_12k.zip
    ```

    Unzip the dataset into `data/datasets/pick_place/`

### Setting up datasets

The code requires the datasets in a `data` folder in the following format:

  ```bash
  habitat-web-baselines/
    data/
      scene_datasets/
        mp3d/
          JeFG25nYj2p
            JeFG25nYj2p.glb
            JeFG25nYj2p.navmesh
            ...
    datasets/
      objectnav/
        objectnav_mp3d_70k/
          train/
      pick_place/
        pick_place_12k/
          train/
  ```

### Test setup

To verify that the data is set up correctly, run:

  ```bash
  python examples/objectnav_replay.py --path data/datasets/objectnav/objectnav_mp3d_70k/sample/sample.json.gz
  ```

## Usage

### Training


For training the behavior cloning policy on the ObjectGoal Navigation task using the environment based setup:
    
1. Use the following script for multi-node training

  ```bash
  sbatch job_scripts/run_objectnav_training.sh habitat_baselines/config/objectnav/il_ddp_objectnav.yaml
  ```

2. To run training on a single node use:

  ```bash
  sbatch job_scripts/run_objectnav_training.sh habitat_baselines/config/objectnav/il_objectnav.yaml
  ```

For training the behavior cloning policy on the PickPlace task using the disk based setup:
    
1. Use the following script for multi-node training

  ```bash
  sbatch job_scripts/run_pickplace_training.sh ddp
  ```

2. To run training on a single node use:

  ```bash
  sbatch job_scripts/run_pickplace_training.sh single_node
  ```

### Evaluation


For evaluating a checkpoint on the ObjectGoal Navigation task using the environment based setup:
    
1. Use the following script if trained using distributed setup

  ```bash
  sbatch job_scripts/run_objectnav_eval.sh habitat_baselines/config/objectnav/il_ddp_objectnav.yaml
  ```

2. Use the following script for evaluating single node checkpoint

  ```bash
  sbatch job_scripts/run_objectnav_eval.sh habitat_baselines/config/objectnav/il_objectnav.yaml
  ```

For evaluating the behavior cloning policy on the PickPlace task using the disk based setup:
    
1. Use the following script if trained using dristributed setup

  ```bash
  sbatch job_scripts/run_pickplace_eval.sh ddp
  ```

2. Use the following script for evaluating single node checkpoint

  ```bash
  sbatch job_scripts/run_pickplace_eval.sh single_node
  ```

## Citation

If you use this code in your research, please consider citing:

```
@inproceedings{ramrakhya2022,
      title={Habitat-Web: Learning Embodied Object-Search from Human Demonstrations at Scale},
      author={Ram Ramrakhya and Eric Undersander and Dhruv Batra and Abhishek Das},
      year={2022},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
}
```
