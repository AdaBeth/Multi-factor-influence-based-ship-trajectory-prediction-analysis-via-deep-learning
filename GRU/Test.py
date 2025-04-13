import torch.nn as nn
import pandas as pd
import torch
import numpy as np
from dataset import Data_utility, normal_std
from tt import Data_utilitys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

batch_size = 1
criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
criterion = criterion.cuda()
evaluateL1 = evaluateL1.cuda()
evaluateL2 = evaluateL2.cuda()


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def FD(x, y):
    distence = np.sqrt(np.power(x[:, 0] - y[:, 0], 2) + np.power(x[:, 1] - y[:, 1], 2))
    return max(distence)


def AED(x, y):
    distence = np.sqrt(np.power(x[:, 0] - y[:, 0], 2) + np.power(x[:, 1] - y[:, 1], 2))
    return np.mean(distence)


# test_num = [93, 94, 111, 315, 708, 991, 995, 999, 1220]
# test_num = [354336000, 374728000, 412081720, 412427003, 412439059, 413559862, 900404567, 901401525, 999968766]
test_num = [212250000, 309972000, 412429910, 413272610, 413272710, 413505370, 414238000, 414369000, 414400890]

model = torch.load('./model/model_CFD_Bi.pth').to('cpu')
model.eval()

for num in [414400890]:
    num = str(num)
    data = Data_utility(file_name=r'path',
                        train=0.99998, valid=0.00001, cuda=False, horizon=12, window=5, normalize=2)

    pred = None
    test = None
    for X, Y in data.get_batches(data.train[0], data.train[1], 1, False):
        output = model(X).reshape(1, 2)
        if pred is None:
            pred = output
            test = Y
        else:
            pred = torch.cat((pred, output), dim=0)
            test = torch.cat((test, Y))

    pred = pred.data.cpu().numpy()
    truth = data.train[1].numpy()
    for i in range(2):
        truth[:, i] = truth[:, i] * (data.max_[i] - data.min_[i]) + data.min_[i]
        pred[:, i] = pred[:, i] * (data.max_[i] - data.min_[i]) + data.min_[i]

    # print(truth)
    # print(pred)
    truth = truth[:, :2]
    pred = pred[:, :2]
    plt.plot(truth[:, 0], truth[:, 1], label='ground truth')
    plt.plot(pred[:, 0], pred[:, 1], label='prediction')
    plt.legend()
    plt.show()
    print('%.7f' % (mean_squared_error(pred, truth) * 100))
    print('%.6f' % mean_absolute_error(pred, truth))
    print('%.6f' % smape(pred, truth))  # 57.76942355889724
    print('%.6f' % np.sqrt(pow(pred[-1, 0] - truth[-1, 0], 2) + pow(pred[-1, 1] - truth[-1, 1], 2)))
    print('%.6f' % FD(pred, truth))
    print('%.6f' % AED(pred, truth))

    df = pd.DataFrame({'lon': pred[:, 0], 'lat': pred[:, 1]})
    df.to_csv(r'filepath')
