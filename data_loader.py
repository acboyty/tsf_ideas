import numpy as np 
import os


def data_loader(data_dir, lookback=30, lookahead=7):
    data_fp = os.path.join(data_dir, 'with_holiday.npy')
    data = np.load(data_fp)
    
    # gen batch
    X, y = [], []
    for st in range(data.shape[0]):
        if st + lookback + lookahead > data.shape[0]:
            break
        X.append(data[st:st + lookback])
        y.append(data[st + lookback:st + lookback + lookahead, 0])
    X, y = np.array(X), np.array(y)
    
    return X, y
