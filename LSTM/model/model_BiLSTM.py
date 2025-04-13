import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn


class BiLSTM(Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.rnn = LSTM(config.input_size, config.hidden_size, bidirectional=True, batch_first=True,
                        num_layers=config.lstm_layers, dropout=config.dropout_rate)
        self.linear = Linear(config.hidden_size * 2, config.output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y),
                              batch_size=config.batch_size)

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = BiLSTM(config).to(device)
    if config.add_train:
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
        # model = torch.load(config.model_save_path + config.model_name)
    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()
        train_loss_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()
            pred_Y = model(_train_X)

            if not config.do_continue_train:
                hidden_train = None
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()
                hidden_train = (h_0, c_0)
            loss = criterion(pred_Y, _train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = model(_valid_X)
            if not config.do_continue_train: hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur) +
                    'lr:{:.6f}.'.format(config.learning_rate))
        if config.do_train_visualized:
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model, config.model_save_path + config.model_name)

        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:
                # logger.info(" The training stops early in epoch {}".format(epoch))
                config.learning_rate = config.learning_rate * 0.1
                if config.learning_rate < 0.00001:
                    torch.save(model, config.model_save_path + config.model_name)
                    break
                bad_epoch = 0

        if epoch + 1 == 200:
            torch.save(model, config.model_save_path + config.model_name)


def predict(config, test_X):
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = BiLSTM(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    result = torch.Tensor().to(device)

    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()
