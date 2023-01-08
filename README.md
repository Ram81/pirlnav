# PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav

Code for our paper [PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav]().  [Project Page]()

Ram Ramrakhya, Dhruv Batra, Erik Wijmans, Abhishek Das


## Installation

1. Run the following commands:

```
git clone https://github.com/Ram81/pirlnav.git
git submodule update --init

conda create -n pirlnav python=3.7 cmake=3.14.0

cd habitat-sim/
pip install -r requirements.txt
./build.sh --headless

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

cd habitat-lab/
pip install -r requirements.txt

pip install -e habitat-lab
pip install -e habitat-baselines

pip install -e .
```


## Data

### Downloading HM3D Scene Dataset

- Download the HM3D dataset using the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) (download the full HM3D dataset for use with habitat)

- Move the HM3D scene dataset or create a symlink at `data/scene_datasets/hm3d`.


### Downloading Human Demonstrations Dataset

You can use the following datasets to reproduce results reported in our paper.

| Dataset| Scene dataset | Split | Link | Extract path |
| ----- | --- | --- | --- | --- |
| ObjectNav-HD | HM3D | 77k | [objectnav_hm3d_hd_70k.json.gz]() | `data/datasets/objectnav/objectnav_hm3d_hd_70k/` |
| ObjectNav-SP | HM3D | 240k | [objectnav_hm3d_sp_240k.json.gz]() | `data/datasets/objectnav/objectnav_hm3d_sp_240k/` |
| ObjectNav-FE | HM3D | 70k | [objectnav_hm3d_fe_240k.json.gz]() | `data/datasets/objectnav/objectnav_hm3d_fe_70k/` |

The demonstration datasets released as part of this project are licensed under a [Creative Commons Attribution-NonCommercial 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

### Dataset Folder Structure

The code requires the datasets in `data` folder in the following format:

  ```bash
  ├── habitat-web-baselines/
  │  ├── data
  │  │  ├── scene_datasets/
  │  │  │  ├── hm3d/
  │  │  │  │  ├── JeFG25nYj2p.glb
  │  │  │  │  └── JeFG25nYj2p.navmesh
  │  │  ├── datasets
  │  │  │  ├── objectnav/
  │  │  │  │  ├── hm3d/
  │  │  │  │  │  ├── v1/
  │  │  │  │  │  |   ├── train/
  │  │  │  │  │  |   ├── val/
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

For training the behavior cloning policy on the Pick-and-Place task using the disk based setup:
    
1. Use the following script for multi-node training

  ```bash
  sbatch job_scripts/run_pickplace_training.sh ddp
  ```

2. To run training on a single node use:

  ```bash
  sbatch job_scripts/run_pickplace_training.sh single_node
  ```

### Evaluation

To evaluate pretrained checkpoint on ObjectGoal Navigation, download the `objectnav_mp3d_v1` dataset from [here](https://github.com/facebookresearch/habitat-lab#task-datasets).

For evaluating a checkpoint on the ObjectGoal Navigation task using the environment based setup:
    
1. Use the following script if trained using distributed setup

  ```bash
  sbatch job_scripts/run_objectnav_eval.sh habitat_baselines/config/objectnav/il_ddp_objectnav.yaml data/datasets/objectnav_mp3d_v1 /path/to/checkpoint
  ```

2. Use the following script for evaluating single node checkpoint

  ```bash
  sbatch job_scripts/run_objectnav_eval.sh habitat_baselines/config/objectnav/il_objectnav.yaml data/datasets/objectnav_mp3d_v1 /path/to/checkpoint
  ```

For evaluating the behavior cloning policy on the Pick-and-Place task using the disk based setup:
    
1. Use the following script if trained using dristributed setup

  ```bash
  sbatch job_scripts/run_pickplace_eval.sh ddp
  ```

2. Use the following script for evaluating single node checkpoint

  ```bash
  sbatch job_scripts/run_pickplace_eval.sh single_node
  ```

## Reproducing Results

We provide best checkpoints for agents trained on ObjectNav task with imitation learning and RL finetuning. You can use the following checkpoints to reproduce results reported in our paper.

| Task | Split | Checkpoint | Success Rate | SPL |
| --- | --- | --- | --- | --- |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | v1 | [objectnav_il.ckpt]() | 64.1 | 27.1 |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | v1 | [objectnav_rl_ft.ckpt]() | 70.4 | - |


## Citation

If you use this code in your research, please consider citing:

```
@article{pirlnav_rramrakhya2023,
      title={PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav},
      author={Ram Ramrakhya and Dhruv Batra and Erik Wijmans and Abhishek Das},
      year={2022},
}
```

