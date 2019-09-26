from utils.score_functions import r2, mae, mse, explained_variance, max_error, msle, median_absolute_error


class Model(object):
    """模型"""

    def fit(self, *args, **kwargs):
        """"""

    def predict(self, *args, **kwargs):
        """"""

    def score(self, *args, **kwargs):
        """"""


class Classifier(Model):
    """分类器"""


class Regressor(Model):
    """回归器"""

    def score(self, X_true, Y_true, scorer='r2'):
        """ 计算回归器的评分

        Parameters
        ----------
        X_true：样本, (n_samples, n_features)
        Y_true：样本值, (n_samples, )
        scorer：评分指标，默认为'r2'，另可取mse、mae、ev、me、msle、medae
        Returns
        -------
        模型在X_true, Y_true上的评分

        """
        Y_predict = self.predict(X_true)

        score_function_dict = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'ev': explained_variance,
            'me': max_error,
            'msle': msle,
            'medae': median_absolute_error
        }

        try:
            score_function = score_function_dict.get(scorer)
        except Exception:
            raise NameError(
                "%s not in the given list(r2、mse、mae、ev、me、msle、medae)" % scorer)
        return score_function(Y_true, Y_predict)


class Clusterer(Model):
    """聚类器"""
