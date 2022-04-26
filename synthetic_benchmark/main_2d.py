import torch.nn as nn
from torch.utils.data import DataLoader
from loss import *
import copy
from utils import *

# =========== CONSTANTS ==============
# Training
NUM_EPOCHS = 10000
PRINT_FREQ = NUM_EPOCHS // 10
BATCH_SIZE = 256
NUM_TRAIN_ITERS = 4
NUM_VAL_ITERS = 1
NUM_TRAIN_SAMPLES = BATCH_SIZE * NUM_TRAIN_ITERS
NUM_VAL_SAMPLES = BATCH_SIZE * NUM_VAL_ITERS
NUM_TEST_SAMPLES = BATCH_SIZE * NUM_VAL_ITERS

# Dimensions
X_DIM = 2
Y_DIM = 2

# Data Range
Y_UB = torch.ones(Y_DIM) * 5
Y_LB = torch.ones(Y_DIM) * -5

# Linear Relation and Noise Scale
NOISE_SIGMA = 1.
NOISE_COVARIANCE = torch.eye(Y_DIM) * (NOISE_SIGMA ** 2)
ORACLE_MATRIX = torch.randn([X_DIM, Y_DIM]) * 0.01

# Normal Distribution Parameters
Y_COVARIANCE = torch.eye(Y_DIM)
Y_COVARIANCE = Y_COVARIANCE * 0.5 + torch.ones_like(Y_COVARIANCE) * 0.5
Y_MEAN = (Y_LB + Y_UB) / 2
# Specify which training distribution to use
TRAIN_DIST = 'normal'

# predefine distributions
DIST_DICT = {
    'uniform': torch.distributions.Uniform(Y_LB, Y_UB),
    'normal': torch.distributions.MultivariateNormal(loc=Y_MEAN, covariance_matrix=Y_COVARIANCE)
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

def f(x):
    # This function will never be called, so we leave the inverse here
    y = ORACLE_MATRIX.inverse() @ x.unsqueeze(-1)
    return y.squeeze()


def f_inv(y):
    x = ORACLE_MATRIX @ y.unsqueeze(-1)
    return x.squeeze()


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
    y_train = DIST_DICT[TRAIN_DIST].sample((NUM_TRAIN_SAMPLES,))
    assert len(y_train) == NUM_TRAIN_SAMPLES

    # Assume a gaussian noise has been added to observed y
    noise_distribution = torch.distributions.MultivariateNormal(torch.zeros(Y_DIM), covariance_matrix=NOISE_COVARIANCE)
    noise = noise_distribution.sample((NUM_TRAIN_SAMPLES,))

    # then the oracle y should be
    y_train_oracle = y_train - noise

    x_train = f_inv(y_train_oracle)

    # Evaluate on balanced (uniform) y distribution
    y_eval = DIST_DICT['uniform'].sample((NUM_VAL_SAMPLES,))
    x_eval = f_inv(y_eval)

    # Test set
    y_test = DIST_DICT['uniform'].sample((NUM_TEST_SAMPLES,))
    x_test = f_inv(y_test)

    train_loader = DataLoader(DummyDataset(x_train, y_train), BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(DummyDataset(x_eval, y_eval), BATCH_SIZE)
    test_loader = DataLoader(DummyDataset(x_test, y_test), BATCH_SIZE)

    return train_loader, eval_loader, test_loader


def prepare_model():
    model = LinearModel(input_dim=X_DIM, output_dim=Y_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
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
    gmm = get_gmm(dist=DIST_DICT[TRAIN_DIST], n_components=1)
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
            optimizer.add_param_group({'params': criterion.parameters(), 'lr': 0.01})
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
    visualize_md(models_trained, train_loader, test_loader, Y_LB, Y_UB)


if __name__ == '__main__':
    main()
