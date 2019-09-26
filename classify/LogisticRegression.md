许多问题的预测结果是一个在连续空间的数值，比如房价预测问题，可以用线性模型来描述：
$$y=w_1x_1+w_2x_2+...+w_nx_n+b$$
但也有很多场景需要输出的是概率估算值，例如：根据邮件内容判断是垃圾邮件的可能性；根据医学影像判断肿瘤是恶性的可能性；手写数字分别是0、1、2、3、4、5、6、7、8、9的可能性。这些都是分类问题。
要预测各类别的概率值，需要将预测输出值控制在[0，1]区间内。对于二元分类问题，它的目标是正确预测两个可能的标签中的一个，就像判断是不是垃圾邮件，答案只有是或否两种。此时可考虑⽤逻辑回归（Logistic Regression）来处理这类问题。
逻辑回归也叫回归，但本质上它能更好的处理分类问题。

如何保证输出值始终落在0和1之间呢？
这里有⼀个⾮常好的函数叫Sigmoid函数（S型函数），它的输出值正好具有以上特性。其定义如下：
$$g(z)=\frac 1 {1+e^{-z}}\tag{1}$$


通过sigmoid函数，可以把线性模型的输出映射到0到1之间的概率，从而实现⼆元分类的思想。

# 逻辑回归中的损失函数
在模型优化时需要计算损失函数，线性回归的损失函数是MSE均方差损失函数，即平方损失，如果把逻辑回归的损失函数也定义为平方损失，会得到如下函数：

$$J(w)=\frac 1 n\sum_{i=1}^n(g(z_i)-y_i)^2$$

这个损失函数是一个⾮凸的函数，是有多个极⼩值的。

![1](https://i.loli.net/2019/08/04/OLJDy8gF7p4dEw5.png)

⽤梯度下降优化算法就很有可能使优化过程陷⼊到局部最优的局部极⼩值中去， 使得函数不能达到全局最优。所以建议大家在逻辑回归中不要采⽤平⽅损失函数。

那这里应该采⽤什么样的损失函数呢？
对于二元逻辑回归的损失函数，一般采用对数损失函数，其定义如下:
$$J(w,b)=\sum(-ylog(\hat y)-(1-y)log(1-\hat y))$$

其中$\hat y$是预测值

为什么要采用这样的函数呢？
假设样本$(x,y)$中的标签值$y=1$，那么理想的预测结果$\hat y=1$。$y$与$\hat y$越接近，损失就越⼩。如果$\hat y=1$，上图公式的后半部分为0，整个损失函数就变成了$-log(\hat y)$。根据对数函数的性质，当$\hat y$越接近1，$-log(\hat y)$的值越⼩的。反之，如果标签值$y=0$，上图公式的前半部分为0，整个损失函数就变成了$-log(1-\hat y)$，当$\hat y$越⼩时，损失值越⼩。
通过这样的对数损失函数，就能较好的刻画预测值和标签值之间的损失关系，⽽且这个损失函数是凸函数：
![2](https://i.loli.net/2019/08/05/4j85z3ukdXAsclJ.png)

# 多元分类和softmax回归
之前已经提到逻辑回归可生成介于0和1之间的小数。例如，某电子邮件分类器的逻辑回归输出值为0.8，表明电子邮件是垃圾邮件的概率为80%，不是垃圾邮件的概率为20%。很明显，这封电子邮件是垃圾邮件与不是垃圾邮件的概率之和为1。
在处理多元分类中，Softmax将逻辑回归的思想延伸到多类别领域。
在多类别问题中，Softmax为每个类别分配一个小数形式的概率，介于0到1之间，并且这些概率的和必须是1。

**Softmax** 层实际上是通过Softmax方程来实现，把y的值经过运算，映射到多分类问题中属于每个类别的概率值：
其计算公式如下：
$$p_i=\frac{e^{(y_i)}}{\sum_{k=1}^ce^{y_k}}$$

这里的$y_k$指的是所有的类别



**交叉熵**
$$H(p,q)=-\sum_xp(x)log(q(x))$$
刻画的是两个概率分布之间的距离，p代表正确答案，q代表的是预测值，交叉熵越小，两个概率的分布约接近，损失越低。对于机器学习中的多分类问题，通常用交叉熵做为损失函数。

下面来看一个交叉熵计算的例子：

假设有一个3分类问题，某个样例的正确答案是（1，0，0），即它属于第一个类别。甲模型经过softmax回归之后的预测答案是（0.5，0.2，0.3），乙模型经过softmax回归之后的预测答案是（0.7，0.1，0.2）。它们俩哪一个模型预测的更好一些呢（更接近正确答案）？

通过下面交叉熵的计算可以看到，乙模型的预测更好：

![1](https://i.loli.net/2019/08/05/uhkoWgr8SOLe7aI.png)

于是，多分类的损失函数为：
$$Loss=-\frac 1 n \sum_{i=1}^ny_ilog(\hat y_i)$$
# logostic分布

设X是连续随机变量，X服从logistic分布是指X具有以下**分布函数**和**密度函数**：
$$F(x)=P(X\le X)=\frac 1 {1+e^{-(x-\mu)/\gamma}}$$
$$f(x)=F'(x)=\frac {e^{-(x-\mu)/\gamma}}{\gamma (1+e^{-(x-\mu)/\gamma})^2}$$
式中，$\mu$是位置参数，$\gamma > 0$是形状参数



**逻辑函数可视化**：

```python
import matplotlib.pyplot as plt
import numpy as np
import math
e = math.e
x = np.linspace(-10,10,1e6)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
```
![1](https://i.loli.net/2019/03/13/5c885785089fd.png)

**假设函数**：$$h_{\theta}(x)=g(\theta^TX)=\frac 1 {1+e^{-\theta^TX}}\tag{2}$$

**代价函数**：$$J(\theta)=\frac 1 m \sum_{i=1}^mCost(h_{\theta}(x^{(i)}),y^{(i)}) \tag{3}$$

当$y=1$时：
$Cost(h_{\theta}(x^{(i)}),y^{(i)}) =  -log(h_{\theta}(x)) \tag{4-1}$

当$y=0$时：
$Cost(h_{\theta}(x^{(i)}),y^{(i)}) =  -log(1-h_{\theta}(x)) \tag{4-2}$



$$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}\tag{5}$$


其函数图像为：


![](https://img-blog.csdnimg.cn/20190124223104140.jpg)



从图中可以看出:
- $y=1$，当预测值$h_{\theta}(x)=1$时，代价函数$Cost$的值为0，这是我们想要的（模型预测完全正确时，代价达到最小）。当预测值离1越远，其代价函数越大，这也是我们想要的
- 同理$y=0$，当预测值=0时，代价函数达到最小值；预测值离0越远，其代价函数越大。

**代价函数推导过程(采用极大似然估计)**



假设函数$h_{\theta}(x)$表示预测结果为1的概率，则：
$$P(y=1|x;\theta)=h_{\theta}(x)\\
P(y=0|x;\theta)=1-h_{\theta}(x)\tag{6}$$

将公式6合并为一个公式：
$$P(y|x;\theta)=h_{\theta}(x)^y*(1-h_{\theta}(x))^{1-y}\tag{7}$$

取似然函数：
$$L(\theta)=\prod_{i=1}^mP(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^m(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}\tag{8}$$

对数似然函数：
$$l_{\theta}=log(L(\theta))=\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}\tag{9}$$

最大似然估计是取使得似然函数最大化的$\theta$，令损失函数$J(\theta)=-\frac 1 m l(\theta)$，则最大化的$l(\theta)$即为最小化的$J(\theta)$



**梯度下降法迭代公式**：
$$\theta_j=\theta_j-\alpha(\frac \partial {\partial \theta_j})J(\theta)=\theta_j-\alpha \frac 1 m \sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}\tag{10}$$

**矩阵形式**：
$$\theta=\theta-\alpha \frac 1 m x^T(g(x\theta)-y)\tag{11}$$

**推导如下**：

![](https://img-blog.csdnimg.cn/20190124223156308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpbGlnZXkx,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20190124223207611.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpbGlnZXkx,size_16,color_FFFFFF,t_70)

**带惩罚项的逻辑回归**



$$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}\tag{12}$$ <!--_-->

重复直至收敛：

   ${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$

   ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$

$j=1,2,...n$


# SoftMax函数
SoftMax函数是将各个类别的打分转化为合适的概率值的函数。
这个函数要满足以下几个条件：
- 打分越高概率越高
- 概率值$p\in [0, 1]$
- 所有的类别的概率之和=1

$Soft(a, b, c) = (\frac {e^a}{e^a+e^b+e^c},\frac {e^b}{e^a+e^b+e^c},\frac {e^c}{e^a+e^b+e^c})$

# SoftMax回归
假设$X$是单个样本的特征，$W$和$b$是SoftMax模型的参数。
SoftMax模型的第一步就是计算各个类别的logit：
$Logit=W^TX+b$
