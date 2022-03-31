# NYUD2-DIR
## Installation

#### Prerequisites

1. Download and extract NYU v2 dataset to folder `./data` using

```bash
python download_nyud2.py
```

2. __(Optional)__ We have provided required meta files `nyu2_train_FDS_subset.csv` and `test_balanced_mask.npy`  for efficient FDS feature statistics computation and balanced test set mask in folder `./data`. To reproduce the results in the paper, please directly use these two files. If you want to try different FDS computation subsets and balanced test set masks, you can run

```bash
python preprocess_nyud2.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- numpy, pandas, scipy, tqdm, matplotlib, PIL, gdown, tensorboardX


## Getting Started

#### Train a model using Balanced MSE

#### GAI

```bash
# preprocess gmm
python preprocess_gmm.py

python train.py \
--bmse --imp gai --gmm gmm.pkl --init_noise_sigma 1.0 --fix_noise_sigma
```

#### BNI
```bash
python train.py \
--bmse --imp bni --init_noise_sigma 1.0 --fix_noise_sigma
```

#### Evaluate a trained checkpoint

```bash
python test.py --eval_model <path_to_evaluation_ckpt>
```

## Reproduced Benchmarks and Model Zoo

We provide below reproduced results on NYUD2-DIR (metric `RMSE`).

| Model | Overall | Many-Shot | Medium-Shot | Few-Shot | Download |
|:-----:|:-------:|:---------:|:-----------:|:--------:| :------: |
|  GAI  |  1.279  |   0.819   |    0.917    |  1.705   | [model](https://drive.google.com/file/d/15VDffeOphB-mBlI7TRP0s-_AP3u1imWk/view?usp=sharing) |
|  BNI  |  1.281  |   0.833   |    0.856    |  1.714   | [model](https://drive.google.com/file/d/1xRqgEi5rjrX2qqalBVtyFT2abShxgmgU/view?usp=sharing) |
