import torch.nn as nn
from torch.utils.data import DataLoader

from loss import *
import copy
from utils import *

# =========== CONSTANTS ==============
# Training
NUM_EPOCHS = 2000
PRINT_FREQ = NUM_EPOCHS // 5
BATCH_SIZE = 256
NUM_TRAIN_ITERS = 1024 // BATCH_SIZE
NUM_VAL_ITERS = 1024 // BATCH_SIZE
NUM_TRAIN_SAMPLES = BATCH_SIZE * NUM_TRAIN_ITERS
NUM_VAL_SAMPLES = BATCH_SIZE * NUM_VAL_ITERS
NUM_TEST_SAMPLES = BATCH_SIZE * NUM_VAL_ITERS

# Data Range
Y_UB = 10
Y_LB = 0

# Linear Relation and Noise Scale
K = 1
B = 0
NOISE_SIGMA = 1.0

# Normal Distribution Parameters
Y_MEAN = (Y_LB + Y_UB) / 2
Y_SIGMA = 0.5  # High imbalance
# Y_SIGMA = 0.75  # Medium imbalance
# Y_SIGMA = 1.    # Low imbalance

# Exponential Distribution Parameters
EXP_RATE = 2.  # High imbalance
# EXP_RATE = 1.5  # Medium imbalance
# EXP_RATE = 1.   # Low imbalance

# Specify which training distribution to use
TRAIN_DIST = 'normal'
# TRAIN_DIST = 'exp'

# predefine distributions
DIST_DICT = {
    'uniform': torch.distributions.Uniform(Y_LB, Y_UB),
    'normal': torch.distributions.Normal(loc=Y_MEAN, scale=Y_SIGMA),
    'exp': torch.distributions.Exponential(EXP_RATE),
}

CRITERIA_TO_USE = [
    'MSE',
    'Reweight',
    'GAI',
    'BMC',
    'GAI Learnable Noise',
    'BMC Learnable Noise'
]


# ======= END OF CONSTANTS ==========


# Define a linear regressor
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


def prepare_data():
    # Training label samples
    y_train = DIST_DICT[TRAIN_DIST].sample((2 * NUM_TRAIN_SAMPLES, 1))
    y_train = y_train[y_train >= Y_LB]  # trim
    y_train = y_train[y_train <= Y_UB]  # trim
    y_train = y_train[:NUM_TRAIN_SAMPLES].unsqueeze(-1)
    assert len(y_train) == NUM_TRAIN_SAMPLES

    # Assume a gaussian noise has been added to observed y
    noise_distribution = torch.distributions.Normal(loc=0, scale=NOISE_SIGMA)
    noise = noise_distribution.sample((NUM_TRAIN_SAMPLES, 1))

    # then the oracle y should be
    y_train_oracle = y_train - noise

    x_train = (y_train_oracle - B) / K

    # Evaluate on balanced (uniform) y distribution
    y_eval = DIST_DICT['uniform'].sample((NUM_VAL_SAMPLES, 1))
    x_eval = (y_eval - B) / K

    # Test set
    y_test = DIST_DICT['uniform'].sample((NUM_TEST_SAMPLES, 1))
    x_test = (y_test - B) / K

    train_loader = DataLoader(DummyDataset(x_train, y_train), BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(DummyDataset(x_eval, y_eval), BATCH_SIZE)
    test_loader = DataLoader(DummyDataset(x_test, y_test), BATCH_SIZE)

    return train_loader, eval_loader, test_loader


def prepare_model():
    model = LinearModel(input_dim=1, output_dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return model, optimizer, scheduler


def train(train_loader, eval_loader, test_loader, model, optimizer, scheduler, criterion):
    best_eval_loss = 1e8
    model_best = None
    for epoch in range(NUM_EPOCHS):
        train_loss = AverageMeter('train loss')
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            train_loss.update(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % PRINT_FREQ == 0:
            print('epoch: ', epoch + 1)
            model.eval()
            eval_loss = AverageMeter('eval loss')
            for data, target in eval_loader:
                pred = model(data)
                loss = F.mse_loss(pred, target)
                eval_loss.update(loss.item())

            print(train_loss)
            print(eval_loss)
            print('-' * 10)
            if best_eval_loss > eval_loss.avg:
                model_best = copy.deepcopy(model)
                best_eval_loss = eval_loss.avg

    print('best eval loss {:.6f}'.format(best_eval_loss))
    model_best.eval()
    test_loss = AverageMeter('test loss')
    for data, target in test_loader:
        pred = model(data)
        loss = F.mse_loss(pred, target)
        test_loss.update(loss.item())
    print(test_loss)
    print('=' * 20)
    return model_best, test_loss.avg


def train_model(train_loader, eval_loader, test_loader):
    gmm = get_gmm(dist=DIST_DICT[TRAIN_DIST], n_components=1 if TRAIN_DIST == 'normal' else 64)
    criteria = {
        'MSE': nn.MSELoss(),
        'Reweight': ReweightL2(DIST_DICT[TRAIN_DIST]),
        'GAI': GAILossMD(init_noise_sigma=NOISE_SIGMA, gmm=gmm),
        'BMC': BMCLossMD(init_noise_sigma=NOISE_SIGMA),
        # For learnable noise, we assume we don't know the ground truth noise scale
        # Therefore we multiply an offset 1.5 to the ground truth noise scale
        'GAI Learnable Noise': GAILossMD(init_noise_sigma=1.5 * NOISE_SIGMA, gmm=gmm),
        'BMC Learnable Noise': BMCLossMD(init_noise_sigma=1.5 * NOISE_SIGMA),
    }
    criteria = {k: criteria[k] for k in CRITERIA_TO_USE}  # Only use selected criteria

    perf_stats = {}
    models_trained = {}

    for criterion_name, criterion in criteria.items():
        print("Training with distribution {} and criterion {}".format(TRAIN_DIST, criterion_name))
        model, optimizer, scheduler = prepare_model()
        if 'Learnable Noise' in criterion_name:
            optimizer.add_param_group({'params': criterion.parameters(), 'lr': 0.001})
        model_best, perf_stats[criterion_name] = \
            train(train_loader, eval_loader, test_loader, model, optimizer, scheduler, criterion)
        models_trained[criterion_name] = model_best

    print('Final results')
    for method in perf_stats:
        print('{0: <20}: {1:.6f}'.format(method, perf_stats[method]))
    return models_trained


def main():
    train_loader, eval_loader, test_loader = prepare_data()
    models_trained = train_model(train_loader, eval_loader, test_loader)
    visualize(models_trained, train_loader, test_loader, Y_LB, Y_UB, K, B)


if __name__ == '__main__':
    main()
