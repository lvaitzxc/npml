[TOC]



# step函数
公式：
$$step(x)=\begin{cases}
1, &x\ge0 \\
0,&x<0
\end{cases}$$

激活函数 Step 更倾向于理论而不是实际，它模仿了生物神经元要么全有要么全无的属性。它无法应用于神经网络，因为其导数是 0（除了零点导数无定义以外），这意味着基于梯度的优化方法并不可行

# sign函数

符号函数

公式：
$$sign(x)=\begin{cases} 
1,&x>0 \\
0,&x=0 \\
-1,&x<0
\end{cases}$$

图像：

![1](https://i.imgur.com/JpEAlvz.png)

# sigmoid函数
公式：
$$sigmoid(z)=\frac 1 {1+e^{-z}}$$

图像：
![1](https://i.imgur.com/5U2FqdB.png)

优缺点：
- 它能够把输入的连续实值变换为0和1之间的输出，特别的，如果是非常大的负数，那么输出就是0；如果是非常大的正数，输出就是1.
- 在深度神经网络中梯度反向传递时导致梯度爆炸和梯度消失，其中梯度爆炸发生的概率非常小，而梯度消失发生的概率比较大
- Sigmoid 的 output 不是0均值。这是不可取的，因为这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入。 产生的一个结果就是：如$x>0, f=w^Tx+b$, 那么对w求局部梯度则都为正，这样在反向传播的过程中w要么都往正方向更新，要么都往负方向更新，导致有一种捆绑的效果，使得收敛缓慢。 当然了，如果按batch去训练，那么那个batch可能得到不同的信号，所以这个问题还是可以缓解一下的。因此，非0均值这个问题虽然会产生一些不好的影响，不过跟上面提到的梯度消失问题相比还是要好很多的。
- 其解析式中含有幂运算，计算机求解时相对来讲比较耗时。对于规模比较大的深度网络，这会较大地增加训练时间。

# tanh函数
公式：$$tanh(x)=\frac {e^x-e^{-x}}{e^x+e^{-x}}$$
图像（右图为其导数的图像）：
![1](https://i.imgur.com/T5TGzLE.png)


优缺点：
- 解决了sigmoid函数非0均值的问题
- 仍无法解决梯度消失、幂运算的问题

# relu函数

公式：
$$relu(x)=max(0,x)$$

图像：
![1](https://i.imgur.com/buYAhtl.png)

优缺点：
- 解决了梯度消失的问题（在正区间）
- 计算速度快，只需计算是否大于0
- 收敛速度远快于sigmoid和tanh
- relu的输出不是0均值
- dead relu problem，指某些神经元可能永远不会激活，导致相应的参数永远不会更新。这主要有两个原因导致：
    - 非常不幸的参数初始化，这种情况比较少见
    - 学习率太高导致在训练过程中参数更新太大，不幸使网络进入这种状态
- 尽管存在这两个问题，relu目前仍是最常用的activation function，在搭建人工神经网络的时候推荐优先尝试

# prelu函数
公式：
$$leaky\_relu(x)=max(ax,x)$$
图像：
![1](https://i.imgur.com/WQOsomK.png)

优缺点：
- 为了解决Dead ReLU Problem，提出了将ReLU的前半段设为$ \alpha x$而非0
- 理论上来讲，PReLU有ReLU的所有优点，外加不会有Dead ReLU问题，但是在实际操作当中，并没有完全证明PReLU总是好于ReLU

当$\alpha=0.01$时，即为leaky relu函数
# ELU (Exponential Linear Units) 函数
公式：
 $$f(x)=\begin {cases}
 x,&x>0 \\
 \alpha(e^x-1),&otherwise
 \end{cases}$$
图像：
![1](https://i.imgur.com/QJJzw8z.png)

 
 优缺点：
 - 不会有dead relu问题
 - 输出均值接近于0
 - 计算量稍大
 - 类似于Leaky ReLU，理论上虽然好于ReLU，但在实际使用中目前并没有好的证据ELU总是优于ReLU

# maxout函数
 
 这个函数可以参考论文《maxout networks》，Maxout是深度学习网络中的一层网络，就像池化层、卷积层一样等，我们可以把maxout 看成是网络的激活函数层，我们假设网络某一层的输入特征向量为：$X=（x_1,x_2,...,x_d）$，也就是我们输入是d个神经元。Maxout隐藏层每个神经元的计算公式如下：

公式：
$$h_i(x)=max_{j \in [1, k]} x^TW_{...ij}+b_{ij}$$

其中k人为给出，$W_{d*m*k}$和$b_{m*k}$为需要学习的参数

如果我们设定参数k=1，那么这个时候，网络就类似于以前我们所学普通的MLP网络。
我们可以这么理解，本来传统的MLP算法在第i层到第i+1层，参数只有一组，然而现在我们不这么干了，我们在这一层同时训练n组的w、b参数，然后选择激活值Z最大的作为下一层神经元的激活值，这个max（z）函数即充当了激活函数。

# swish函数
自控门激活函数

公式：
$$swish(x)=\frac x {1+e^{-x}}$$

图像：
![1](https://i.imgur.com/ryS9vxr.png)

优缺点：
- Swish 激活函数的性能优于 ReLU 函数
- 根据上图，我们可以观察到在 x 轴的负区域曲线的形状与 ReLU 激活函数不同，因此，Swish 激活函数的输出可能下降，即使在输入值增大的情况下。大多数激活函数是单调的，即输入值增大的情况下，输出值不可能下降。而 Swish 函数为 0 时具备单侧有界（one-sided boundedness）的特性，它是平滑、非单调的。更改一行代码再来查看它的性能，似乎也挺有意思

# 待补充 TODO 
https://dashee87.github.io/data%20science/deep%20learning/visualising-activation-functions-in-neural-networks/
https://www.jiqizhixin.com/articles/2017-10-10-3