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


### Download Demonstrations Dataset

You can use the following datasets to reproduce results reported in our paper.

| Dataset| Scene dataset | Split | Link | Extract path |
| ----- | --- | --- | --- | --- |
| ObjectNav-HD | HM3D | 77k | [objectnav_hm3d_hd_70k.json.gz]() | `data/datasets/objectnav/objectnav_hm3d_hd_70k/` |
| ObjectNav-SP | HM3D | 240k | [objectnav_hm3d_sp_240k.json.gz]() | `data/datasets/objectnav/objectnav_hm3d_sp_240k/` |
| ObjectNav-FE | HM3D | 70k | [objectnav_hm3d_fe_240k.json.gz]() | `data/datasets/objectnav/objectnav_hm3d_fe_70k/` |

The demonstration datasets released as part of this project are licensed under a [Creative Commons Attribution-NonCommercial 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

### Download HM3D Episode Dataset

- Download the ObjectNav HM3D episode dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets).


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


For training the behavior cloning policy on the ObjectGoal Navigation task:
    
1. Use the following script for multi-node training

  ```bash
  sbatch scripts/1-objectnav-il.sh
  ```

For RL finetuning the behavior cloned policy on the ObjectGoal Navigation task:
    
1. Use the following script for multi-node training

  ```bash
  sbatch scripts/2-objectnav-rl-ft.sh
  ```

### Evaluation

To evaluate pretrained checkpoint on ObjectGoal Navigation, download the `objectnav_hm3d_v1` dataset from [here](https://github.com/facebookresearch/habitat-lab#task-datasets).

For evaluating a checkpoint on the ObjectGoal Navigation task using behavior cloning checkpoint:
    
1. Use the following command

  ```bash
  sbatch scripts/1-objectnav-il-eval.sh /path/to/checkpoint
  ```

For evaluating a checkpoint on the ObjectGoal Navigation task using RL finetuned checkpoint:

1. Use the following command

  ```bash
  sbatch scripts/1-objectnav-rl-ft-eval.sh/path/to/checkpoint
  ```


## Reproducing Results

We provide best checkpoints for agents trained on ObjectNav task with imitation learning and RL finetuning. You can use the following checkpoints to reproduce results reported in our paper.

| Task | Checkpoint | Success Rate | SPL |
| --- | --- | --- | --- | --- |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | [objectnav_il.ckpt]() | 64.1 | 27.1 |
| 🆕[ObjectNav](https://arxiv.org/abs/2006.13171) | [objectnav_rl_ft.ckpt]() | 70.4 | - |


## Citation

If you use this code in your research, please consider citing:

```
@article{pirlnav_rramrakhya2023,
      title={PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav},
      author={Ram Ramrakhya and Dhruv Batra and Erik Wijmans and Abhishek Das},
      year={2022},
}
```

