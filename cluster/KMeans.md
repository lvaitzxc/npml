代价函数：
$$J(c,u)=\sum_{i=1}^k||x(i)-u_{c^{(i)}}||^2$$
$u_{c^{(i)}}$表示第i个类的均值。
我们希望代价函数最小，直观的来说，各类内的样本越相似，其与该类均值间的误差平方越小，对所有类所得到的误差平方求和，即可验证分为k类时，各聚类是否是最优的

# 算法流程

输入：待聚类的样本X、预先给定的聚类数K
输出：样本X中每个样本被分到的类、最终的所有聚类中心
流程：
1. 初始化K个聚类中心作为最初的中心
2. 循环每个样本，计算其与K个聚类中心的距离，将该样本分到距离最小的那个聚类中心
3. 将每个聚类中的样本均值作为新的聚类中心
4. 重复步骤2和3直到聚类中心不再变化

# 性能分析
- 优点
  - 是解决聚类问题的一种经典算法，简单、快速
  - 对处理大数据集，该算法是相对可伸缩和高效率的。它的复杂度是$O(nkt)$,其中, $n$是所有对象的数目, $k$ 是簇的数目, $t$ 是迭代的次数。通常$k<<n$ 且$t<<n$
  - 当结果簇是密集的，而簇与簇之间区别明显时, 它的效果较好
- 缺点
  - 在簇的平均值被定义的情况下才能使用，这对于处理符号属性的数据不适用
  - 必须事先给出K
  - 对初值敏感，对于不同的初始值，可能会导致不同结果
  - 对躁声和孤立点数据是敏感的，少量的该类数据能够对平均值产生极大的影响。

# 如何确认聚类数K TODO