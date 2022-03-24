# Balanced MSE
Code for the paper:

**[Balanced MSE for Imbalanced Visual Regression](https://arxiv.org/abs/2203.16427)**  
Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Ziwei Liu

CVPR 2022 (**Oral**)

<div align="left">
  <img src="figures/intro.png" width="500px" />
</div>


## Live Demo

Check out our [live demo](https://huggingface.co/spaces/jiawei011/Demo-Balanced-MSE) in the Hugging Face :hugs: space!

<div align="left">
  <img src="figures/regress.gif" width="300px" />
</div>

## Tutorial

We provide a minimal working example of Balanced MSE using the BMC implementation on a small-scale dataset, 
[Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). 

<p class="aligncenter">
    <a href="https://colab.research.google.com/github/jiawei-ren/BalancedMSE/blob/main/tutorial/balanced_mse.ipynb" target="_parent">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a> 
</p>

The notebook is developed on top of [Deep Imbalanced Regression (DIR) Tutorial](https://github.com/YyzHarry/imbalanced-regression/tree/main/tutorial),
we thank the authors for their amazing tutorial!

## Quick Preview
A code snippet of the Balanced MSE loss is shown below.
It is the BMC implementation for one-dimensional imbalanced regression,
which does not require any label prior beforehand.
```python
def bmc_loss(pred, target, noise_var):
    logits = - (pred - target.T).pow(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale

    return loss
```
`noise_var` is a hyper-parameter. `noise_var` can be optionally optimized in training:
```python
class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

criterion = BMCLoss(init_noise_sigma)
optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})

```

## Run Experiments

Please go into the sub-folder to run experiments.

- [IMDB-WIKI-DIR](./imdb-wiki-dir)
- [NYUD2-DIR](./nyud2-dir)
- IHMR (coming soon)

## Citation
```bib
@inproceedings{ren2021bmse,
  title={Balanced MSE for Imbalanced Visual Regression},
  author={Ren, Jiawei and Zhang, Mingyuan and Yu, Cunjun and Liu, Ziwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgment

This work is supported by NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), the National Research Foundation, Singapore under its AI Singapore Programme, and under the RIE2020 Industry Alignment Fund – Industry Collabo- ration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

The code is developed on top of [Delving into Deep Imbalanced Regression](https://github.com/YyzHarry/imbalanced-regression).




