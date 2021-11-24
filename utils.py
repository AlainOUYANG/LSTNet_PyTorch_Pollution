import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PollutionDataset(Dataset):
    def __init__(self, window, horizon, skip, train=False, train_split=0.8):
        super(PollutionDataset, self).__init__()
        self.data = np.loadtxt('./pollution.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3, 4, 6, 7, 8],dtype=np.float32)
        self.data_train = self.data[:int(self.data.shape[0] * train_split), :]
        self.data_test = self.data[int(self.data.shape[0] * train_split):, :]
        self.window = window
        self.horizon = horizon
        self.skip = skip

        self.data_norm, self.scaler = self.normalization(self.data_train if train else self.data_test)

        self.x, self.y = self._convert_supervise(self.data_norm)

        self.n_samples = len(self.x) - self.window * self.skip - 1

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample

    def __len__(self):
        return self.n_samples

    @staticmethod
    def normalization(data):
        scaler = np.arange(2 * data.shape[1])
        scaler = scaler.reshape(data.shape[1], 2)
        dat = np.zeros(data.shape)
        for i in range(data.shape[1]):
            lst = data[:, i]
            lst_low, lst_high = np.min(lst), np.max(lst)
            scaler[i, 0] = lst_low
            scaler[i, 1] = lst_high
            delta = lst_high - lst_low
            if delta != 0:
                for j in range(data.shape[0]):
                    dat[j, i] = (data[j, i] - lst_low) / delta
        return dat, scaler

    def _convert_supervise(self, data_norm):
        x, y = [], []
        for i in range(self.window * self.skip, len(data_norm) - 11):
            x.append(data_norm[(i - self.window):i, :])
            y.append(data_norm[i, 0:1])
            # y.append(data_norm[i:i + self.horizon, 0:1])  # For now, just perform a 1 step forecasting
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class Args:
    def __init__(self, window=30, skip=5, highway_window=3, horizon=10,
                 batch_size=64, hidRNN=64, hidCNN=48, hidSkip=64, CNN_kernel=6,
                 dropout=0.2, optimizer='adam', output_fun='sigmoid',
                 cuda=False):
        self.window = window
        self.skip = skip
        self.highway_window = highway_window
        self.horizon = horizon
        self.batch_size = batch_size

        self.hidRNN = hidRNN
        self.hidCNN = hidCNN
        self.hidSkip = hidSkip
        self.CNN_kernel = CNN_kernel

        self.dropout = dropout
        self.optimizer = optimizer
        self.output_fun = output_fun

        self.cuda = cuda
        self.log = './logs/train.log'
