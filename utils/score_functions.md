[TOC]

# 分类指标
目前的分类指标只计算二分类 正样本为1 负样本为0

TP：正确预测为正类的样本数
FP：错误预测为正类的样本数
TN：正确预测为负类的样本数
FN：错误预测为负类的样本数

## 准确率
$$accuracy = \frac{(TP + TN)} { (TP + TN + FP + FN)} $$
## 精确率

被预测为正类中有多少是真的正类的
$$precision = \frac {TP} { (TP + FP)}$$
## 召回率
正类中有多少比例是正确预测了的
$$recall = \frac {TP}  {(TP + FN)}$$

## f1
$$f1 = \frac {2 * precision * recall}  {(precision + recall)}$$

## kappa系数
$$k=\dfrac{p_o-p_e}{1-p_e}$$
其中：
$$p_o=\frac {正确分类总数}{总样本数}$$
假设每一类的真实样本个数分别为$a_1,a_2,...,a_C$, 预测出来的每一类的样本个数分别为$b_1,b_2,...,b_C$
$$p_e=\dfrac{a_1\times b_1+a_2\times b_2+...+a_C\times b_C}{n\times n} $$

举例说明：

预测类别\实际类别|A|B|C
----|----|----|----
A|239|21|16
B|16|73|4
C|6|9|280
$$p_o=\dfrac{239+73+280}{239+21+16+16+73+4+6+9+280}=0.8916$$

$$p_e=\dfrac{(239+16+6)\times (239+21+16)+(21+73+9)\times (16+73+4) +(16+4+280)\times (6+9+280)}{664\times 664}=0.3883$$
则
$$k=\dfrac{0.8916-0.3883}{1-0.3883}=0.8228$$

## Jaccard相似度
$$J(y_i, \hat{y}_i) = \frac{|y_i \cap \hat{y}_i|}{|y_i \cup \hat{y}_i|}$$

## Hinge损失
https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss TODO
# 回归指标
## R方
   $$ R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
   
  

## 可解释方差

可解释变异（英语：explained variation）在统计学中是指给定数据中的变异能被数学模型所解释的部分。通常会用方差来量化变异，故又称为可解释方差（explained variance）。

除可解释变异外，总变异的剩余部分被称为未解释变异（unexplained variation）或残差（residual）。

$$explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$$

## 最大误差
$$ \text{Max Error}(y, \hat{y}) = max(| y_i - \hat{y}_i |)$$
## 平均绝对误差
$$ \text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$$
## 均方误差
$$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2$$

## 平均对数误差
mean squared logarithmic error

当具有指数增长的目标时，例如人口数量，商品在一段时间内的平均销售额等，此度量最好用。请注意，此度量标准会对低于预测的估计值进行惩罚。

$$\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2$$

## 中位绝对误差

median absolute error

median_absolute_error特别有趣，因为它对异常值有很强的鲁棒性。通过获取目标和预测之间的所有绝对差异的中值来计算损失。

$$\text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid)$$
# 聚类指标