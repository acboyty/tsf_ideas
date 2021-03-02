import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from net import RNNNet, RNNGATENet
import numpy as np 
import os


class RNNModel():
    def __init__(self, lookback, lookahead, input_dim, hid_dim, device, data_dir, task):
        self.lookback = lookback
        self.lookahead = lookahead
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
        self.data_dir = data_dir
        self.task = task

        self.net = RNNNet(self.input_dim, self.hid_dim, self.lookahead, self.lookback).to(self.device)

    def fit(self, X_train, y_train, X_val, y_val, metric, max_epoch, patience, batch_size, lr, weight_decay):
        # data loader
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True)

        # optimizer
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        # metric
        if metric == 'mse':
            criterion = nn.MSELoss().to(self.device)
        elif metric == 'mae':
            criterion = nn.L1Loss().to(self.device)

        # train & validation -> early stopping
        best_epoch, best_loss = -1, np.inf
        for epoch in range(max_epoch):
            self.net.train()
            for idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                forecast = self.net(X_batch)
                loss = criterion(y_batch, forecast)

                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                forecast = self.net(X_val)
                loss = criterion(y_val, forecast).item()
                if loss < best_loss:
                    if epoch % 50 == 0:
                        print(f'epoch {epoch:04d} loss: {loss:7.5f} is the best epoch')
                    best_epoch, best_loss = epoch, loss
                    torch.save({
                        'net_state_dict': self.net.state_dict(),
                        'best_epoch': best_epoch,
                    }, os.path.join(self.data_dir, self.task + '.RNNNet.th'))
                else:
                    if epoch % 50 == 0:
                        print(f'epoch {epoch:04d} loss: {loss:7.5f} no improvement for {epoch - best_epoch} epoch(es)')
                    if epoch - best_epoch > patience:
                        break
    
    def eval(self, X_test):
        checkpoint = torch.load(os.path.join(self.data_dir, self.task + '.RNNNet.th'))
        self.net.load_state_dict(checkpoint['net_state_dict'])
        X_test = torch.Tensor(X_test)
        self.net.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            forecast = self.net(X_test.to(self.device))
        return forecast.cpu().numpy()


class RNNGATEModel():
    def __init__(self, lookback, lookahead, hid_dim, device, data_dir, task):
        self.lookback = lookback
        self.lookahead = lookahead
        self.hid_dim = hid_dim
        self.device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
        self.data_dir = data_dir
        self.task = task

        self.net = RNNGATENet(self.lookback, self.hid_dim, self.lookahead).to(self.device)

    def fit(self, X_train, y_train, X_val, y_val, metric, max_epoch, patience, batch_size, lr, weight_decay):
        # data loader
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True)

        # optimizer
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        # metric
        if metric == 'mse':
            criterion = nn.MSELoss().to(self.device)
        elif metric == 'mae':
            criterion = nn.L1Loss().to(self.device)

        # train & validation -> early stopping
        best_epoch, best_loss = -1, np.inf
        for epoch in range(max_epoch):
            self.net.train()
            for idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                forecast = self.net(X_batch)
                loss = criterion(y_batch, forecast)

                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                forecast = self.net(X_val)
                loss = criterion(y_val, forecast).item()
                if loss < best_loss:
                    if epoch % 50 == 0:
                        print(f'epoch {epoch:04d} loss: {loss:7.5f} is the best epoch')
                    best_epoch, best_loss = epoch, loss
                    torch.save({
                        'net_state_dict': self.net.state_dict(),
                        'best_epoch': best_epoch,
                    }, os.path.join(self.data_dir, self.task + '.RNNGATE.th'))
                else:
                    if epoch % 50 == 0:
                        print(f'epoch {epoch:04d} loss: {loss:7.5f} no improvement for {epoch - best_epoch} epoch(es)')
                    if epoch - best_epoch > patience:
                        break
    
    def eval(self, X_test):
        checkpoint = torch.load(os.path.join(self.data_dir, self.task + '.RNNGATE.th'))
        self.net.load_state_dict(checkpoint['net_state_dict'])
        X_test = torch.Tensor(X_test)
        self.net.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            forecast = self.net(X_test.to(self.device))
        return forecast.cpu().numpy()