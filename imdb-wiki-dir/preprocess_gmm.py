from datasets import IMDBWIKI
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os
import time
import joblib
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
# Default args
# training/optimization related
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=90, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')

parser.add_argument('--reweight', type=str, default='none', choices=['none', 'inverse', 'sqrt_inv'],
                    help='cost-sensitive reweighting scheme')
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

# Args for GMM
parser.add_argument('--K', type=int, default=8, help='GMM number of components')


def prerpocess_gmm():
    args = parser.parse_args()
    # Data
    end_time = time.time()
    print('Getting Train Loader...')
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train = df[df['split'] == 'train']
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train',
                             reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    print(time.time() - end_time)
    end_time = time.time()
    print('Training Loader Done.')
    print('Curate training labels...')
    all_labels = []
    for _, (_, targets, _) in tqdm(enumerate(train_loader)):
        all_labels.append(targets)
    all_labels = torch.cat(all_labels).reshape(1, -1)
    print('All labels shape: ', all_labels.shape)
    print(time.time() - end_time)
    end_time = time.time()
    print('Training labels curated')
    print('Fitting GMM...')
    gmm = GaussianMixture(n_components=args.K, random_state=0, verbose=2).fit(
        all_labels.reshape(-1, 1).cpu().numpy())
    print(time.time() - end_time)
    end_time = time.time()
    print('GMM fiited')
    print("Dumping...")
    gmm_dict = {}
    gmm_dict['means'] = gmm.means_
    gmm_dict['weights'] = gmm.weights_
    gmm_dict['variances'] = gmm.covariances_
    gmm_path = 'gmm.pkl'
    joblib.dump(gmm_dict, gmm_path)
    print(time.time() - end_time)
    print('Dumped at {}'.format(gmm_path))


if __name__ == '__main__':
    prerpocess_gmm()
