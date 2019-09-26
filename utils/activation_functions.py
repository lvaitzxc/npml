"""激活函数 TODO"""
import numpy as np


def elu(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.array([i if i > 0 else alpha * (np.exp(i) - 1) for i in x])


def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.array([max(0.01 * i, i) for i in x])


def prelu(x: np.ndarray, alpha=0.5) -> np.ndarray:
    return np.array([max(alpha * i, i) for i in x])


def relu(x: np.ndarray) -> np.ndarray:
    return np.array([max(0, i) for i in x])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sign(x: np.ndarray) -> np.ndarray:
    """符号函数"""
    return np.sign(x)  # 这个有点尴尬😅


def swish(x: np.ndarray) -> np.ndarray:
    """自控门激活函数"""
    return x / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))