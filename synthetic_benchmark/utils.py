import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from sklearn.mixture import GaussianMixture


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class DummyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


def get_gmm(dist, n_components):
    # fit a **ground truth** label distribution
    all_labels = dist.sample([10000, ])     # assume sufficient samples
    if len(all_labels.shape) == 1:
        all_labels = all_labels.unsqueeze(-1)
    gmm = GaussianMixture(n_components=n_components).fit(all_labels)
    gmm_dict = {'means': gmm.means_, 'weights': gmm.weights_, 'variances': gmm.covariances_}
    return gmm_dict


def make_dataframe(x, y, method=None):
    x = list(x[:, 0].detach().numpy())
    y = list(y[:, 0].detach().numpy())
    if method is not None:
        method = [method for _ in range(len(x))]
        df = pd.DataFrame({'x': x, 'y': y, 'Method': method})
    else:
        df = pd.DataFrame({'x': x, 'y': y})
    return df


def unzip_dataloader(training_loader):
    all_x = []
    all_y = []
    for data, label in training_loader:
        all_x.append(data)
        all_y.append(label)
    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)
    return all_x, all_y


def visualize(model_dict, train_loader, test_loader, Y_LB, Y_UB, K, B):
    sns.set_theme(palette='colorblind')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Get model outputs
    model_df = []
    x_test, _ = unzip_dataloader(test_loader)
    for model_name in model_dict:
        model = model_dict[model_name]
        model.eval()
        y = model(x_test)
        model_df.append(make_dataframe(x_test, y, model_name))

    training_df = make_dataframe(*unzip_dataloader(train_loader), 'Training')
    test_df = make_dataframe(*unzip_dataloader(test_loader), 'Testing')
    oracle_df = make_dataframe(*unzip_dataloader(test_loader), 'Oracle')

    # plot oracle and predictions
    sns.lineplot(data=pd.concat([oracle_df, *model_df], ignore_index=True), x='x', y='y', hue='Method', ax=ax1)

    # plot data points
    sns.scatterplot(data=training_df, x='x', y='y', color='#003ea1', alpha=0.2, linewidths=0, s=100, ax=ax1,
                    legend=False)

    ax1.set_xlim((Y_LB - B) / K, (Y_UB - B) / K)
    ax1.set_ylim(Y_LB, Y_UB)
    ax1.set_xlabel(r'$x$', fontsize=10)
    ax1.set_ylabel(r'$y$', fontsize=10)

    # plot training histogram
    bins = np.linspace(Y_LB, Y_UB, 20)
    sns.histplot(data=training_df, y='y', kde=False, stat='density', hue='Method', common_norm=False, bins=bins, ax=ax2)

    # plot kdeplot
    sns.kdeplot(data=pd.concat([training_df, *model_df, test_df], ignore_index=True), y='y', hue='Method',
                common_norm=False, ax=ax2)

    ax2.set_ylim(Y_LB, Y_UB)
    ax2.set_xlabel(r'$p(y)$', fontsize=10)
    ax2.set_ylabel(r'$y$', fontsize=10)
    plt.tight_layout()
    plt.show()


def hist_3d(ax, data, title, Y_LB, Y_UB, zmax=0.06):
    xx, yy = np.mgrid[Y_LB[0].item():Y_UB[0].item():100j, Y_LB[1].item():Y_UB[1].item():100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = data.transpose(0, 1).detach().cpu().numpy()
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_zlabel('p(y)')
    ax.set_zlim(0, zmax)
    ax.set_title(title)
    surf.set_clim(vmin=0, vmax=zmax)
    ax.view_init(55, 25)


def visualize_md(model_dict, train_loader, test_loader, Y_LB, Y_UB):
    num_models = len(list(model_dict.keys()))
    fig = plt.figure(figsize=((num_models + 2) * 4, 5))
    subplot_idx = 1

    # train distribution
    ax = fig.add_subplot(1, num_models + 2, subplot_idx, projection='3d')
    subplot_idx += 1
    hist_3d(ax, unzip_dataloader(train_loader)[1], 'Train', Y_LB, Y_UB, zmax=0.14)

    # test distribution
    ax = fig.add_subplot(1, num_models + 2, subplot_idx, projection='3d')
    subplot_idx += 1
    hist_3d(ax, unzip_dataloader(test_loader)[1], 'Test', Y_LB, Y_UB)

    for model_name in model_dict:
        model = model_dict[model_name]
        model.eval()
        x_test, _ = unzip_dataloader(test_loader)
        pred = model(x_test)
        ax = fig.add_subplot(1, num_models + 2, subplot_idx, projection='3d')
        subplot_idx += 1
        hist_3d(ax, pred, model_name, Y_LB, Y_UB)

    plt.tight_layout()
    plt.show()
