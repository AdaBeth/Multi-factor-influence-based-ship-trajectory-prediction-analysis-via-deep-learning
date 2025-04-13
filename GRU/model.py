import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(Model, self).__init__()
        self.use_cuda = True
        self.P = window
        self.m = n_val
        self.hidR = hidRNN
        self.GRU = nn.GRU(self.m, self.hidR)

        self.linear = nn.Linear(self.hidR, self.m)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        _, h = self.GRU(x1)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(h)  # res: [batch, n_val]
        return res


class BiGRU(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(BiGRU, self).__init__()
        self.use_cuda = True
        self.P = window
        self.m = n_val
        self.hidR = hidRNN
        self.GRU = nn.GRU(self.m, self.hidR, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(self.hidR * 2, self.m)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        # x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        out, h = self.GRU(x)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(out)  # res: [batch, n_val]
        return torch.squeeze(res[:, -1, :])