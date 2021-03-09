import numpy as np 
import torch
import torch.nn as nn
from data_loader import data_loader
from sklearn.model_selection import train_test_split
from model import RNNModel, RNNGATEModel
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import os


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.set_deterministic(True)


DATA_DIR = './data/sin_mul_0.1/'

USE_HOLIDAY = 'no_use'  # no_use, feature, gate

LOOKBACK = 40
LOOKAHEAD = 1

SEED = 2021
seed_everything(SEED)


X, y = data_loader(data_dir=DATA_DIR, lookback=LOOKBACK, lookahead=LOOKAHEAD)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
print(np.sum(np.abs(X_train[:, 1] - 1) > 1e-5), np.sum(np.abs(X_val[:, 1] - 1) > 1e-5), np.sum(np.abs(X_test[:, 1] - 1) > 1e-5))


if USE_HOLIDAY == 'no_use':
    X_train[:, :, [1]] = 0
    X_val[:, :, [1]] = 0
    X_test[:, :, [1]] = 0

    model = RNNModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, input_dim=2, hid_dim=20, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val, 
        metric='mae', max_epoch=100000, patience=400, 
        batch_size=64, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test)


elif USE_HOLIDAY == 'feature':
    model = RNNModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, input_dim=2, hid_dim=20, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val, 
        metric='mae', max_epoch=100000, patience=400, 
        batch_size=64, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test)


elif USE_HOLIDAY == 'gate':
    model = RNNGATEModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, hid_dim=20, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val, 
        metric='mae', max_epoch=100000, patience=400, 
        batch_size=64, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test)


print(f'MAE {mean_absolute_error(y_test, forecast):.5f}', end=' ')
print(f'MSE {mean_squared_error(y_test, forecast):.5f}')

np.save(os.path.join(DATA_DIR, 'y_test'), y_test)
np.save(os.path.join(DATA_DIR, USE_HOLIDAY + '.forecast'), forecast)
