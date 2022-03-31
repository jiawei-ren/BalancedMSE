# IMDB-WIKI-DIR
## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. __(Optional)__ We have provided required IMDB-WIKI-DIR meta file `imdb_wiki.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget


## Getting Started

#### Stage 1. Train the base model
Train a vanilla model as the base model: 
```bash
python train.py
```
Alternatively, a pretrained base model can be downloaded from [here](https://drive.google.com/file/d/1-wHqT7T3-6QDCVQi-2UdqU_R0PkokSvM/view?usp=sharing).

#### Stage 2. Train a model using RRT + Balanced MSE

#### GAI

```bash
# preprocess gmm
python preprocess_gmm.py

python train.py \
--balanced_metric \
--bmse --imp gai --init_noise_sigma 10. --gmm gmm.pkl \
--sigma_lr 0.01 \
--retrain_fc --pretrained <path_to_base_model_ckpt> \
--lr 0.0001 --epoch 10
```

##### BMC

```bash
python train.py \
--balanced_metric \
--bmse --imp bmc --init_noise_sigma 10. \
--retrain_fc --pretrained <path_to_base_model_ckpt> \
--lr 0.0001 --epoch 10
```

##### BNI

```bash
python train.py \
--balanced_metric \
--bmse --imp bni --init_noise_sigma 10. \
--sigma_lr 0.01 \
--retrain_fc --pretrained <path_to_base_model_ckpt>  \
--lr 0.0001 --epoch 10
```

#### Evaluate a trained checkpoint

```bash
python train.py [...evaluation model arguments...] --evaluate --resume <path_to_evaluation_ckpt>
```

## Reproduced Benchmarks and Model Zoo

We provide below reproduced results on IMDB-WIKI-DIR (base method `Vanilla`, metric `MAE`).


| Model | Overall | Many-Shot | Medium-Shot | Few-Shot | Download  |
|:-----:|:-------:|:---------:|:-----------:|:--------:|:---------:|
|Vanilla| 13.923  |   7.323   |   15.925    |  32.778  | [model](https://drive.google.com/file/d/1-wHqT7T3-6QDCVQi-2UdqU_R0PkokSvM/view?usp=sharing) |
|  GAI  | 12.690  |   7.589   |   12.880    |  28.307  | [model](https://drive.google.com/file/d/1GzA2Hn1BSSZ46Tottd3ro0DcAoUEm4qu/view?usp=sharing) |
|  BMC  | 12.654  |   7.649   |   12.689    |  28.097  | [model](https://drive.google.com/file/d/1PDinGFE2XZmKKlrld2nzJqYY1_VuFkS6/view?usp=sharing) |
|  BNI  | 12.650  |   7.647   |   12.696    |  28.077  | [model](https://drive.google.com/file/d/16HcDjP6-EK80QvZMuKI05bbiAlvUwdGE/view?usp=sharing) |