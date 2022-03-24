import loaddata
import argparse
from sklearn.mixture import GaussianMixture
import torch
import joblib
import time
from loaddata import TRAIN_BUCKET_NUM

parser = argparse.ArgumentParser(description='')

# Args for GMM
parser.add_argument('--K', type=int, default=16, help='GMM number of components')

bucket_centers = torch.linspace(0, 10, 101)[:-1] + 0.05
TRAIN_BUCKET_NUM = [TRAIN_BUCKET_NUM[7]] * 7 + TRAIN_BUCKET_NUM[7:]

def fit_gmm(args):
    end_time = time.time()
    all_labels = []
    # There are too many pixels in NYUD2-DIR to fit a GMM
    # We directly use the statistics provided in the original code
    for i in range(100):
        all_labels += [bucket_centers[i] for _ in range(TRAIN_BUCKET_NUM[i] // 1000000)]
    all_labels = torch.tensor(all_labels).reshape(1, -1)
    print('All labels shape: ', all_labels.shape)
    print(time.time() - end_time)
    end_time = time.time()
    print('Training labels curated')
    print('Fitting GMM...')
    gmm = GaussianMixture(n_components=args.K, random_state=0, verbose=2).fit(
        all_labels.reshape(-1, 1).numpy())
    print(time.time() - end_time)
    print('GMM fiited')
    print("Dumping...")
    gmm_dict = {}
    gmm_dict['means'] = gmm.means_
    gmm_dict['weights'] = gmm.weights_
    gmm_dict['variances'] = gmm.covariances_
    return gmm_dict

def main():
    args = parser.parse_args()
    train_loader = loaddata.getTrainingData(args, args.batch_size)
    gmm_dict = fit_gmm(train_loader, args)
    gmm_path = 'gmm.pkl'
    joblib.dump(gmm_dict, gmm_path)
    print('Dumped at {}'.format(gmm_path))


if __name__ == '__main__':
    main()
