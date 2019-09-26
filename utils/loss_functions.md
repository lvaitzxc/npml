
# 0-1损失函数

预测值和目标值不相等为1，否则为0
# 绝对值损失函数
$$l = |y-\hat y|$$
#  $l_1$损失函数
$$l=\sum_{i=1}^n|Y_i-f(x_i)|$$
# $l_2$损失函数
$$l=\sum_{i=1}^n(Y_i-f(x_i))^2$$
# 平方损失函数
回归问题中经常使用平方损失函数
$$l= \sum_{i=1}^n(Y_i-f(x_i))^2$$

# 对数损失函数
逻辑回归使用该函数
$$l(y, p(y|x)) = - log p(y|x) $$

# 交叉熵损失函数
$$H(p,q)=-\sum_xp(x)log(q(x))$$

# Hinge损失函数


TODO https://www.cnblogs.com/luxiao/p/5783017.html



https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss