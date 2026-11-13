# Advanced Visual AI coursework

doesn't do anything

## Setup

First clone this repo. Then: 

### Install dependencies with Conda

Conda works on the lab machines, so that's why I'm using it. The dependencies are in requirements.txt so you can install them another way if you'd like

In the project root, run: 

```bash
conda create -n advai python=3.11 pip
conda activate advai
python -m pip install -r requirements.txt
```

And then you can enter and leave the environment with `conda activate advai`and `conda deactivate`.


### Download the dataset

Download the DIV2K training and validation HR images and the NTIRE 2018 Track 1
bicubic x8 LR images from https://data.vision.ee.ethz.ch/cvl/DIV2K/.
Extract them into a directory called data:
(i should make a script to do this more easily)

```text
data/
├── DIV2K_train_HR/
├── DIV2K_train_LR_x8/
├── DIV2K_valid_HR/
└── DIV2K_valid_LR_x8/
```

### explanation of how to run it goes here




