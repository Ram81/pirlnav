# Not all demonstrations are created equal


## Installation

1. Clone the repository and install `il_rl_baselines` using the commands below. Note that `python=3.6` is required for working with `il_rl_baselines`. All the development was done on `habitat-lab=0.1.6`.

    ```bash
    git clone https://github.com/Ram81/il_rl_baselines.git
    cd il_rl_baselines

    # We require python>=3.6 and cmake>=3.10
    conda create -n habitat-web python=3.6 cmake=3.14.0
    conda activate habitat-web

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

1. For use with [habitat-web-baselines](https://github.com/Ram81/habitat-web-baselines) and your own python code, add habitat-sim to your `PYTHONPATH`. For example modify your `.bashrc` (or `.bash_profile` in Mac OS X) file by adding the line:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/habitat-sim/
   ```


## Citation
If you use this work, please cite using:

```
@misc{ilrl_baselines,
  title = {Not all Demonstrations are Created Equal: An ObjectNav Case Study for Effectively Combining Imitation and Reinforcement Learning},
  author = {Ram Ramrakhya, Erik Wijmans, Dhruv Batra, Abhishek Das},
  howpublished = {\url{https://github.com/Ram81/il_rl_baselines}},
  year = {2022}
}
```
