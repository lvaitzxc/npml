from numpy import ndarray

from metrics.classification import accuracy
from metrics.regression import r2


class Model(object):
    """所有模型的基类"""
    _model_type = "model"

    def fit(self, *args, **kwargs):
        """训练"""

    def predict(self, *args, **kwargs):
        """预测"""

    def score(self, *args, **kwargs):
        """评分"""

    def plot(self, *args, **kwargs):
        """绘图"""

    @property
    def model_type(self):
        return self._model_type


class SupervisedModel(Model):
    """监督模型"""
    _model_type = "supervised_model"


class UnsupervisedModel(Model):
    """无监督模型"""
    _model_type = "unsupervised_model"


class SemiSupervisedModel(Model):
    """半监督模型"""
    _model_type = "semi_supervised_model"


class Regressor(SupervisedModel):
    """回归模型"""
    _model_type = "regressor"

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        return r2(y_true, y_pred)


class Classifier(SupervisedModel):
    """分类模型"""
    _model_type = "classifier"

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        return accuracy(y_true, y_pred)


class Clusterer(UnsupervisedModel):
    """聚类模型"""
    _model_type = "clusterer"

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        # TODO 评估聚类效果
        pass


class DimensionReducer(UnsupervisedModel):
    """降维模型"""
    _model_type = "dimension_reducer"
