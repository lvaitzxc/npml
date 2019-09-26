岭回归在普通最小二乘法的基础上加上了一个$l_2$惩罚项

损失函数：$J\left(\theta \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{[({{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta _{j}^{2}})]}$ <!--_-->


**正规方程**

$\theta=(X'X+\alpha I)^{-1}X'Y$


**梯度下降法**

一般形式：

重复以下步骤 直到收敛:
* [ ] 


${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$

   ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$

$j=1,2,...n$



${\theta_j}:={\theta_j}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}$

$\alpha$是控制模型复杂度的因子，可看做收缩率的大小。$\alpha$越大，收缩率越大，系数对于共线性的鲁棒性更强

矩阵形式：

$$\theta= \theta(1-\alpha \frac \lambda m) -\alpha \frac 1 m \alpha{X}^T({X\theta} -{Y})$$