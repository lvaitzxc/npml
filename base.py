from metrics.classification import accuracy, precision, recall, f1, roc_auc, hinge
from metrics.regression import r2, mae, mse, explained_variance, max_error, msle, median_absolute_error
from types import FunctionType
from typing import Callable
import numpy as np


class Model(object):
    """模型类 本module下所有模型的基类"""

    def predict(self, *args, **kwargs):
        """"""

    def score(self, *args, **kwargs):
        """"""


class SupervisedModel(Model):
    """监督模型"""

    def fit(self, *args, **kwargs):
        """监督和半监督模型才需要训练 无监督不需要"""


class UnsupervisedModel(Model):
    """无监督模型"""


class SemiSupervisedModel(Model):
    """半监督模型"""

    def fit(self, *args, **kwargs):
        """"""


class BaseMixin:
    """所有Mixin类的基类"""

    def score(self, X: np.ndarray, y: np.ndarray, scorer: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """模型在X,y上的评分

        @param X: 样本数据
        @param y: 样本值
        @param scorer: 评分函数 从metrics
        @return: 评分
        """
        y_predict = self.predict(X)
        score = scorer(y, y_predict)
        return score


class ClassifierMixin:
    """所有分类模型的Mixin类"""
    _model_type = 'Classifier'

    def score(self, X, y, scorer='accuracy'):
        score_function_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'hinge': hinge
        }
        if not score_function_dict.has_key(scorer):
            raise NameError(
                "%s not in the given list(r2、mse、mae、ev、me、msle、medae)" % scorer)
        scorer_function = score_function_dict.get(scorer)
        y_predict = self.predict(X)  # 预测值
        score = scorer_function(y, y_predict)

        return score


class RegressorMixin:
    """所有回归模型的Mixin类"""
    _model_type = 'Regressor'

    def score(self, X, y, scorer='r2'):
        """计算回归器的评分
        @param X: 样本数据
        @param y: 样本值
        @param scorer: 评分函数，可选r2、mse、mae、ev、me、msle、medae
        @return: 模型在数据X,y上的评分
        """
        score_function_dict = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'ev': explained_variance,
            'me': max_error,
            'msle': msle,
            'medae': median_absolute_error
        }
        if not score_function_dict.has_key(scorer):
            raise NameError(
                "%s not in the given list(r2、mse、mae、ev、me、msle、medae)" % scorer)
        scorer_function = score_function_dict.get(scorer)
        y_predict = self.predict(X)  # 预测值
        score = scorer_function(y, y_predict)

        return score


class Clusterer(Model):
    """聚类器"""
