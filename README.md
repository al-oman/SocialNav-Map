# SocialNav-Map: Dynamic Mapping with Human Trajectory Prediction for Zero-Shot Social Navigation
Repository for **SocialNav-Map: Dynamic Mapping with Human Trajectory Prediction for Zero-Shot Social Navigation**

### Getting Started

#### 1. **Preparing conda env**

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
```
conda_env_name=socialnav-map
conda create -n $conda_env_name python=3.9 cmake=3.14.0
conda activate $conda_env_name
```

#### 2. **conda install habitat-sim & habitat-lab**
Following [Habitat-lab](https://github.com/facebookresearch/habitat-lab.git)'s instruction:
```
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat
```

If you encounter network problems, you can manually download the Conda package from [this link](https://anaconda.org/aihabitat/habitat-sim/0.3.1/download/linux-64/habitat-sim-0.3.1-py3.9_headless_bullet_linux_3d6d67d6deae4ab2472cc84df7a3cef1503f606d.tar.bz2) to download the conda bag, and install it via: `conda install --use-local /path/to/xxx.tar.bz2` to download.

Then, assuming you have this repositories cloned (forked from Habitat 3.0), install necessary dependencies of Habitat.
```
cd Falcon
pip install -e habitat-lab
pip install -e habitat-baselines
pip install -r requirements.txt # install other dependencies
```

For other dependencies, see 


#### 3. **Downloading the Social-HM3D & Social-MP3D datasets**

- Download Scene Datasets

Following the instructions for **HM3D** and **MatterPort3D** in Habitat-lab's [Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

- Download Episode Datasets

Download social navigation (SocialNav) episodes for the test scenes, which can be found here: [Link](https://drive.google.com/drive/folders/1V0a8PYeMZimFcHgoJGMMTkvscLhZeKzD?usp=drive_link).

After downloading, unzip and place the datasets in the default location:
```
unzip -d data/datasets/pointnav
```
- Download Leg animation

```
wget https://github.com/facebookresearch/habitat-lab/files/12502177/spot_walking_trajectory.csv -O data/robots/spot_data/spot_walking_trajectory.csv
```

- Download Multi-agent necessary data

```
python -m habitat_sim.utils.datasets_download --uids hab3-episodes habitat_humanoids hab3_bench_assets hab_spot_arm
```

The file structure should look like this:

```
SocialNav-Map/
└── data/
    ├── datasets
    │   └── pointnav
    │       ├── social-hm3d
    │       │   ├── train
    │       │   │   ├── content
    │       │   │   └── train.json.gz
    │       │   └── val
    │       │       ├── content
    │       │       └── val.json.gz
    │       └── social-mp3d
    │           ├── train
    │           │   ├── content
    │           │   └── train.json.gz
    │           └── val
    │               ├── content
    │               └── val.json.gz
    ├── scene_datasets
    ├── robots
    ├── humanoids
    ├── versoned_data
    └── hab3_bench_assets
```

