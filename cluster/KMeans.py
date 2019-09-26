import random

import numpy as np

from base import Clusterer
from utils.distances import euclidean_distance


class KMeans(Clusterer):

    def fit(self, X, k=2, max_iter=100000, tol=1e-10):
        """训练 TODO 计算过程有点问题
        Parameters
        ----------
        X: 训练集, (n_samples, n_features)
        k: 将X聚成k类
        max_iter: 最大迭代次数，到达max_iter则停止迭代
        tol: 质心前后两次变化的最大误差小于tol则停止迭代

        Returns
        ------
        self.centers: 质心, (k)
        self.results: 存放每个点对应到哪个类, key为类，value为点的集合
        """
        # 初始化质心
        global next_results
        self.centers = np.array(random.sample(list(X), k=k))  # 从样本X中随机选取k个点作为初始质心
        # 存放每个点对应到哪个类, key为类，value为点的集合
        self.results = dict(zip(list(range(1, k + 1)), [[] for _ in range(k)]))

        # 开始迭代
        for i in range(max_iter):
            next_results = dict(
                zip(list(range(1, k + 1)), [[] for _ in range(k)]))
            for sample in X:
                index = KMeans._get_nearest_class(sample, self.centers)
                next_results[index].append(sample)
            new_centers = np.array([np.mean(next_results[i], axis=0)
                                    for i in range(1, k + 1)])  # 新的质心

            if np.sum(np.abs(self.centers - new_centers)) <= tol:  # 如果前后两次的质心小于tol则收敛
                print("前后两次质心变化小于tol，视为收敛，停止迭代")
                break
            else:
                self.centers = new_centers
                self.results = next_results

        else:
            print("到达最大迭代次数，停止迭代")
            self.results = next_results

        return self

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
