# [Less Random, More Private: What is the Optimal Subsampling  Scheme for DP-SGD?](https://github.com/ruheyun/yc-blog/issues/7)

## 作者信息

Andy Dong （Stanford University）、 Ayfer Ozgur （Stanford University）

## 名词解释

independent-example mechanisms：每个样本的“是否参与每次训练迭代”的采样过程，是彼此独立生成的。

Balanced Iteration Subsampling (BIS)：本文提出的采样方法，每个样本在T次迭代中恰好参与k次迭代，随机均匀选择，从而消除了样本间参与次数的差异。

## 本文贡献

- 理论上证明了BIS在两个极限情况 ($\sigma \rightarrow 0, \sigma \rightarrow \infty$) 下都可以取得最优。
- 引入了蒙特卡洛审计，消除了先验界中的松弛，进行近似精确的有限噪声核算。
- 大量实验验证了BIS是一个原则性更好的替代泊松子采样的方法。

> 本文的反直觉观点：更多的采样随机性并不一定意味着更强的隐私放大。

## 背景知识

### 差分隐私

Hockey-stick divergence（曲棍球棒散度）：是专门为 DP 的“近似隐私”概念量身定制的等价表述，在差分隐私文献中通常称为
$$E_{\gamma}-divergence$$ 。

以机器学习梯度的视角来定义差分隐私：令 $\chi$ 表示数据域，令 $M$ 表示一个随机算法。考虑置零邻接关系（zero-out adjacency relation）。如果数据集 $D'$ 可以通过将 $D$ 中的一个元素替换为一个总是贡献零梯度的特殊记录 $\perp$ 而得到，则称两个数据集 $D, D' \subseteq \chi$ 是相邻的。

令 $P, Q$ 分别表示 $M(D), M(D')$ 的分布。如果 $P, Q$ 之间的 $e^{\epsilon}$ -hockey-stick divergence）以 $\delta$ 为界，即满足下式，则称机制 $M$ 满足 $(\epsilon, \delta)-DP$ ：
$$H_{e^{\epsilon}}(P \parallel Q) = E_{y \sim P} [max \\{ 1 - e^{\epsilon} \frac{Q(y)}{P(y)}, 0 \\}] = E_{y \sim Q} [max \\{ \frac{P(y)}{Q(y)} - e^{\epsilon}, 0 \\}] \leq \delta$$
