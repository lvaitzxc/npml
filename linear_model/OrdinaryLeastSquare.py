import numpy as np
from numpy.linalg import pinv
from numpy import ndarray
from base import Regressor
from utils.decorators import check_params_type


class OrdinaryLeastSquare(Regressor):
    """普通最小二乘法"""

    def __init__(self, coef_=None, intercept=None):
        self.coef_ = coef_  # 系数
        self.intercept_ = intercept  # 截距项

    @check_params_type()
    def fit(self, X_train: ndarray, y_train: ndarray) -> Regressor:
        """
        @param X_train: 训练数据，二维数组
        @param y_train: X_train的真实值，一维数组
        @return: self
        """
        n_samples, n_features = X_train.shape

        # 给X添加一列1， 将y转换成(n_samples, 1)
        X_train = np.concatenate((np.ones(n_samples).reshape((n_samples, 1)), X_train), axis=1)
        y_train = y_train.reshape((n_samples, 1))

        theta = pinv(X_train.T @ X_train) @ X_train.T @ y_train  # A @ B == np.dot(A, B)

        self.intercept_ = theta[0, 0]  # 截距项
        self.coef_ = theta[1:, 0]  # 系数

        return self

    @check_params_type()
    def predict(self, X_test):
        """
        @param X_test: 测试数据，二维数组
        @return: 测试数据的预测值，一维数组
        """
        return X_test @ self.coef_ + self.intercept_

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        pass

    def plot(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    a = OrdinaryLeastSquare()
    x = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    a.fit(x, y)
    print(a.coef_, a.intercept_)
    assert abs(a.coef_ - 1) <= 0.000001
    assert a.intercept_ <= 0.000001
    b = a.predict(np.array([[10]]))
    print(b)
    assert abs(b[0] - 10) <= 0.000001
