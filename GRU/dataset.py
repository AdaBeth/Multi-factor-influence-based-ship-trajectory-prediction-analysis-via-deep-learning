import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda
        self.P = window
        self.h = horizon
        self.rawdat = np.array(pd.read_csv(file_name, usecols=[2, 3]))  # , usecols=[3, 4, 5, 6], sep=' '
        self.max_ = np.max(self.rawdat, axis=0)
        self.min_ = np.min(self.rawdat, axis=0)
        if (len(self.rawdat.shape)) == 1:
            self.rawdat = self.rawdat.reshape(len(self.rawdat), -1)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.input_size = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.input_size)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.input_size)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.input_size):
                # self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = (self.rawdat[:, i] - self.min_[i]) / (self.max_[i] - self.min_[i])

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.input_size))
        Y = torch.zeros((n, self.input_size))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            # if self.dat[start, 0] == self.dat[end, 0]:
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[end, :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X, Y = inputs[excerpt], targets[excerpt]
            if self.cuda:
                X, Y = X.cuda(), Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size

# data = Data_utility(file_name='all.csv', train=0.8, valid=0.1, cuda=False, horizon=12, window=5, normalize=2)
# data._split(int(0.8 * data.n), int((0.8 + 0.1) * data.n), data.n)
