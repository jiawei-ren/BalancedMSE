import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN


class ReweightL2(_Loss):
    def __init__(self, train_dist, reweight='inverse'):
        super(ReweightL2, self).__init__()
        self.reweight = reweight
        self.train_dist = train_dist

    def forward(self, pred, target):
        reweight = self.reweight
        prob = self.train_dist.log_prob(target).exp().squeeze(-1)
        if reweight == 'inverse':
            inv_prob = prob.pow(-1)
        elif reweight == 'sqrt_inv':
            inv_prob = prob.pow(-0.5)
        else:
            raise NotImplementedError
        inv_prob = inv_prob / inv_prob.sum()
        loss = F.mse_loss(pred, target, reduction='none').sum(-1) * inv_prob
        loss = loss.sum()
        return loss


class GAILossMD(_Loss):
    """
    Multi-Dimension version GAI, compatible with 1-D GAI
    """

    def __init__(self, init_noise_sigma, gmm):
        super(GAILossMD, self).__init__()
        self.gmm = gmm
        self.gmm = {k: torch.tensor(self.gmm[k]) for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss_md(pred, target, self.gmm, noise_var)
        return loss


def gai_loss_md(pred, target, gmm, noise_var):
    I = torch.eye(pred.shape[-1])
    mse_term = -MVN(pred, noise_var*I).log_prob(target)
    balancing_term = MVN(gmm['means'], gmm['variances']+noise_var*I).log_prob(pred.unsqueeze(1)) + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=1)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()


class BMCLossMD(_Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss


def bmc_loss_md(pred, target, noise_var):
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))
    loss = loss * (2 * noise_var).detach()
    return loss
