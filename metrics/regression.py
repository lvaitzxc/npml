"""回归指标"""
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差 mse = ∑(y_true - y_pred)^2 / n"""
    return np.average((y_true - y_pred) ** 2)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """r2"""
    y_mean = y_true.mean()
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_mean)**2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差 mae = ∑|y_true - y_pred| / n"""
    return np.mean(np.abs(y_true - y_pred))


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """可解释方差"""
    return 1 - np.var(y_true - y_pred) / np.var(y_true)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """最大误差"""
    return np.max(np.abs(y_true - y_pred))


def msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均对数误差"""
    return np.mean((np.log(1 + y_true) - np.log(1 + y_pred))**2)


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """"""
