# coding=utf-8
import numpy as np
from sklearn import metrics

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])

# MSE
print(metrics.mean_squared_error(y_true, y_pred))  # 8.107142857142858
# RMSE
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))  # 2.847304489713536
# MAE
print(metrics.mean_absolute_error(y_true, y_pred))  # 1.9285714285714286
# MAPE
print(mape(y_true, y_pred))  # 76.07142857142858，76%
# SMAPE
print(smape(y_true, y_pred))  # 57.76942355889724，58%
