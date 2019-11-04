import random

import numpy as np

from base import ClusterMixin, UnsupervisedModel
from utils.distances import euclidean_distance


class KMeans(UnsupervisedModel, ClusterMixin):

    def fit(self, X, k=8, max_iter=100000, tol=1e-10, distance=euclidean_distance):
        """训练KMeans
        @param X 训练集, (n_samples, n_features)
        @param k 将X聚成k类
        @param max_iter 最大迭代次数，到达max_iter则停止迭代
        @param tol 质心前后两次变化的最大误差小于tol则停止迭代
        @param distance 距离函数
                            - 欧氏距离
                            - 曼哈顿距离
                            - 切比雪夫距离
                            - 闵可夫斯基距离
                            - 标准化欧氏距离
                            - 马氏距离
                            - 巴氏距离
        @return self
                self.centers: 质心, (k)
                self.results: 存放每个点对应到哪个类, key为类，value为点的集合
        """
        # 初始化质心


    @staticmethod
    def _get_nearest_class(sample, centers):
        """点sample离centers中哪个质心更近，返回哪个质心的索引+1"""
        return np.argmin(np.sqrt(np.sum((centers - sample) ** 2, axis=1))) + 1

    def predict(self, X):
        """预测
        Parameters
        ----------
        X: 需要预测的数据集，(n_samples, n_features)

        Returns
        ------
        数据集每个点分到的类, (n_samples)
        """
        return np.array([KMeans._get_nearest_class(x, self.centers) for x in X])
