### Create class to read-in data and process it for forecasting
import os
import torch
from torch.utils.data import random_split

import pandas as pd
import numpy as np
import datetime as dt

from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class WindowDataset(Dataset):
    def __init__(self, path2data: str, input_size: int, target_size: int, epsilon: float=1e-9, mode=None):
        super(WindowDataset, self).__init__()
        df = pd.read_csv(os.path.abspath(path2data), index_col=0)
        df['date'] = df['date'].astype('int')
        df['time_idx'] = [dt.datetime.strptime(str(x)+'-1', "%Y%W-%w") for x in df['date']]
        df = pd.pivot_table(df,
                         values='billings',
                         index='time_idx',
                         columns='product',
                         fill_value=0)
        df = df.iloc[:, np.where(df.std(axis=0))[0]]
        
        X = sliding_window_view(df.iloc[:-target_size], input_size, axis=0).reshape(-1, input_size)
        Y = sliding_window_view(df.iloc[input_size:], target_size, axis=0).reshape(-1, target_size)
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        idxs = np.where(std)[0]
        self.X, self.Y, self.mean, self.std = X[idxs], Y[idxs], mean[idxs], std[idxs]
        self.X = torch.from_numpy((self.X - self.mean) / self.std).float().view(self.X.shape + (1,))
        self.Y = torch.from_numpy((self.Y - self.mean) / self.std).float().view(self.Y.shape + (1,))
        if mode == 'tcn':
            self.X = self.X.view(self.X.shape[0], self.X.shape[2], self.X.shape[1])
            self.Y = self.Y.view(self.Y.shape[0], self.Y.shape[2], self.Y.shape[1])

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]