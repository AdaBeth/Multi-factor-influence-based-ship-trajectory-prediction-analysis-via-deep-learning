import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

ROOT_DIR = r"file"
TRAIN = ROOT_DIR + "Chengshan Jiao data"
VAL = ROOT_DIR + "Chengshan Jiao data"
TEST = ROOT_DIR + "Chengshan Jiao data"
DATASET_PATH = {
    "train": TRAIN,
    "val": VAL,
    "test": TEST
}

obs_len = 5
pre_len = 1


class TrajectoryDataset(Dataset):

    def __init__(self, root_dir, mode):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.sequences = [(self.root_dir / x).absolute() for x in os.listdir(self.root_dir)]
        self.obs_len = obs_len
        self.pre_len = 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = pd.read_csv(self.sequences[idx], usecols=[3, 4])
        # agent_x = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["X"]
        # agent_y = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_x = sequence.iloc[:, 0]
        agent_y = sequence.iloc[:, 1]
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        # return input and target
        train_x, train_y = create_dataset(agent_traj)
        return [train_x[idx], train_y[idx]]


def MinMaxScaler(X):
    min_ = np.array([np.min(X[:, i]) for i in range(X.shape[1])])
    max_ = np.array([np.max(X[:, i]) for i in range(X.shape[1])])
    # print(min_, max_)
    resX = np.empty(shape=X.shape, dtype=float)
    for col in range(X.shape[1]):
        resX[:, col] = (X[:, col] - min_[col]) / (max_[col] - min_[col])
    return resX, min_, max_


def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)


def create_dataset(dataset, look_back=obs_len):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - pre_len):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[(i + look_back):(i + look_back + pre_len)])
    return np.array(dataX), np.array(dataY)


def get_data():
    data = np.array(pd.read_csv('all.csv'))[:1000, :]
    data, min_, max_ = MinMaxScaler(data)
    train_x, train_y = create_dataset(data)
    train_x = train_x.transpose((1, 0, 2))
    train_y = train_y.transpose((1, 0, 2))

    return [torch.Tensor(train_x), torch.Tensor(train_y)]


# train_data = get_data()
# print(train_data[0].shape)

# tra = TrajectoryDataset(DATASET_PATH['train'], 'train')
# data = get_dataset(['train'])
# print(data)
