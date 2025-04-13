import torch.nn as nn
import math
import torch
from model import Model, BiGRU
import numpy as np
from dataset import Data_utility
from optimizer import Optim
import time
import os
import matplotlib.pyplot as plt


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.input_size)
        # loss = criterion(output * scale, Y * scale)
        loss = criterion(output, Y)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.item()
        n_samples += (output.size(0) * data.input_size)
    return total_loss / n_samples


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.input_size)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.input_size)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation, predict


device = "cuda" if torch.cuda.is_available() else "cpu"
data = Data_utility(file_name='./data/all.csv', train=0.9, valid=0.09, cuda=True, horizon=12, window=5, normalize=2)

print(data.train[0].shape, data.train[1].shape)  # torch.Size([10347, 168, 1]) torch.Size([10347, 1])
window = data.train[0].shape[1]
n_val = data.train[0].shape[2]

# model = Model(n_val, window, 128)

model = BiGRU(n_val, window, 128).to(device)
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
criterion = criterion.cuda()
evaluateL1 = evaluateL1.cuda()
evaluateL2 = evaluateL2.cuda()

optimizer = Optim(
    model.parameters(), 'adam', lr=0.0001, max_grad_norm=100, start_decay_at=50, lr_decay=0.5
)

batch_size = 128
epochs = 100
best_val = 1
save = './model/model_CFD_Bi.pth'

print('begin training')

# files = os.listdir(input_path)

for epoch in range(1, epochs):
    # for file in files:
    # data = Data_utility(file_name=input_path + file, train=0.8, valid=0.1, cuda=False, horizon=12, window=5,
    #                     normalize=2)
    epoch_start_time = time.time()
    train_loss = train(data, data.train[0], data.train[1], model, criterion, optimizer, batch_size)
    val_loss, val_rae, val_corr, _ = evaluate(data, data.valid[0], data.valid[1], model, evaluateL2, evaluateL1,
                                              batch_size)
    print(
        '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:7.6f} | valid rse {:7.6f} | valid rae {:7.6f} | valid '
        'corr  {:5.4f} | lr {:5.4f} '
            .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, optimizer.lr))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val:
        with open(save, 'wb') as f:
            torch.save(model, f)
        best_val = val_loss

    if epoch % 100 == 0:
        test_acc, test_rae, test_corr, _ = evaluate(data, data.test[0], data.test[1], model, evaluateL2, evaluateL1,
                                                    batch_size)
        print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

    optimizer.updateLearningRate(val_loss, epoch)

test_acc, test_rae, test_corr, pred = evaluate(data, data.test[0], data.test[1], model, evaluateL2, evaluateL1,
                                               batch_size)
print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

