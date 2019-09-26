"""损失函数"""
import numpy as np


def mean_squared_error_loss(y_true, y_pred):
    """计算均方误差损失函数(最小二乘采用此损失函数)的梯度
    Parameters
    ----------
        y_true: 真实值, (n_samples)
        y_pred: 预测值, (n_samples)
    """
    return np.mean((y_true - y_pred) ** 2)
