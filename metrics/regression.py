"""回归指标"""
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差 mse = ∑(y_true - y_pred) / n"""
    return np.average((y_true - y_pred) ** 2)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """r2"""
    y_mean = y_true.mean()
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_mean)**2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """"""


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """"""


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """"""


def msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """mean squared logarithmic error"""


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """"""
