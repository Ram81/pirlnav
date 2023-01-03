### Installation

1. Run the following commands:

```
git clone git@github.com:Ram81/pirlnav.git
git submodule update --init

conda create -n pirlnav python=3.7 cmake=3.14.0

cd habitat-sim/
pip install -r requirements.txt
./build.sh --headless

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

cd habitat-lab/
pip install -e habitat-lab
pip install -e habitat-baselines

pip install -e .
```