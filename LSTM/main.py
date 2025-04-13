# -*- coding: UTF-8 -*-
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

frame = "pytorch"
if frame == "pytorch":
    from model.model_pytorch import train, predict
elif frame == "keras":
    from model.model_keras import train, predict

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
else:
    raise Exception("Wrong frame seletion")


class Config:
    feature_columns = [3, 4]
    label_columns = [3, 4]
    # label_in_feature_index = [feature_columns.index(i) for i in label_columns]
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1

    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128
    lstm_layers = 2
    dropout_rate = 0.5
    time_step = 5

    do_train = True
    do_predict = True
    add_train = False
    shuffle_train_data = True
    use_cuda = False

    train_data_rate = 0.9
    valid_data_rate = 0.1

    batch_size = 128
    learning_rate = 0.001
    epoch = 250
    patience = 6
    random_seed = 42

    do_continue_train = False
    continue_flag = ""
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    debug_mode = False
    debug_num = 500

    used_frame = frame
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + '_CSJ_Bi' + model_postfix[used_frame]

    train_data_path = r""
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = False
    do_train_visualized = False
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean) / self.std
        # self.norm_data = self.data

        self.start_num_in_test = 0

    def read_data(self):
        if self.config.debug_mode:
            files = os.listdir(self.config.train_data_path)
            all_data_frames = []
            for file in files:
                data_frame = pd.read_csv(self.config.train_data_path + file, index_col=None,
                                         nrows=self.config.debug_num,
                                         usecols=self.config.feature_columns)
                all_data_frames.append(data_frame)
            init_data = pd.concat(all_data_frames, axis=0, ignore_index=True)
            # init_data = pd.read_csv(self.config.train_data_path, sep=' ', nrows=self.config.debug_num,
            #                         usecols=self.config.feature_columns)
        else:
            files = os.listdir(self.config.train_data_path)
            all_data_frames = []
            for file in files:
                data_frame = pd.read_csv(self.config.train_data_path + file, index_col=None,
                                         header=None, usecols=self.config.feature_columns)
                data_frame = data_frame.dropna()
                all_data_frames.append(data_frame)
            init_data = pd.concat(all_data_frames, axis=0, ignore_index=True)
            # print(init_data)
            # init_data = pd.read_csv(self.config.train_data_path, sep=' ', usecols=self.config.feature_columns)
        # print(init_data)
        return init_data.values, init_data.columns.tolist()

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day: self.config.predict_day + self.train_num, self.config.label_in_feature_index]  # 将延后几天的数据作为label

        if not self.config.do_continue_train:
            train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
            train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        else:
            train_x = [
                feature_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [
                label_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval
        test_x = [feature_data[
                  self.start_num_in_test + i * sample_interval: self.start_num_in_test + (i + 1) * sample_interval]
                  for i in range(time_step_size)]
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test:,
                 config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]
    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day]) ** 2, axis=0)
    loss_norm = loss / (origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [x + config.predict_day for x in label_X]

    if not sys.platform.startswith('linux'):
        for i in range(2):  # label_column_num
            logger.info("The predicted {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                        str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                plt.savefig(
                    config.figure_save_path + "{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i],
                                                                                config.used_frame))
        # plt.show()
        print('MSE: %10f' % mean_squared_error(label_data, predict_data))
        print('MAE: %10f' % mean_absolute_error(label_data, predict_data))
        plt.plot(label_data, label_data, label='label', color='b')
        plt.plot(predict_data, predict_data, label='predict', color='r')
        plt.legend()
        plt.show()


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            print(test_X)
            pred_result = predict(config, test_X)
            print(pred_result)
            draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):
        if not key.startswith("_"):
            setattr(con, key, getattr(args, key))

    main(con)
