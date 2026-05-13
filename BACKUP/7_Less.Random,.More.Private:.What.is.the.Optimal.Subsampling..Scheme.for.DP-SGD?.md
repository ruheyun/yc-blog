# [Less Random, More Private: What is the Optimal Subsampling  Scheme for DP-SGD?](https://github.com/ruheyun/yc-blog/issues/7)

## 作者信息

Andy Dong （Stanford University）、 Ayfer Ozgur （Stanford University）

## 名词解释

independent-example mechanisms：每个样本的“是否参与每次训练迭代”的采样过程，是彼此独立生成的。

Balanced Iteration Subsampling (BIS)：本文提出的采样方法，每个样本在T次迭代中恰好参与k次迭代，随机均匀选择，从而消除了样本间参与次数的差异。

Hockey-stick divergence（曲棍球棒散度）：是专门为 DP 的“近似隐私”概念量身定制的等价表述，在差分隐私文献中通常称为 $E_{\gamma}-divergence$ 。

第一性原理：不依赖经验、类比或现有惯例，而是将复杂问题拆解到最基本、不可再分的事实或公理，然后从这些底层基础出发，通过逻辑演绎重新推导解决方案。

布尔超立方体（Boolean Hypercube）：集合 $\\{0, 1\\}^T$ 包含 $2^T$ 个可能得 0、1向量。例如 T=3，包含（0,0,0），（0,0,1），...，（1,1,1）共8个向量。

## 本文贡献

- 理论上证明了BIS在两个极限情况 ($\sigma \rightarrow 0, \sigma \rightarrow \infty$) 下都可以取得最优。
- 引入了蒙特卡洛隐私会计，消除了先验界中的松弛，进行近似精确的有限噪声核算。
- 大量实验验证了BIS是一个原则性更好的替代泊松子采样的方法。

> 本文的反直觉观点：更多的采样随机性并不一定意味着更强的隐私放大。

## 背景知识

### 差分隐私

以机器学习梯度的视角来定义差分隐私：令 $\chi$ 表示数据域，令 $M$ 表示一个随机算法。考虑置零邻接关系（zero-out adjacency relation）。如果数据集 $D'$ 可以通过将 $D$ 中的一个元素替换为一个总是贡献零梯度的特殊记录 $\perp$ 而得到，则称两个数据集 $D, D' \subseteq \chi$ 是相邻的。

令 $P, Q$ 分别表示 $M(D), M(D')$ 的分布。如果 $P, Q$ 之间的 $e^{\epsilon}$ -hockey-stick divergence以 $\delta$ 为界，即满足下式，则称机制 $M$ 满足 $(\epsilon, \delta)-DP$ ：

$$H_{e^{\epsilon}}(P \parallel Q) = E_{y \sim P} [max \\{ 1 - e^{\epsilon} \frac{Q(y)}{P(y)}, 0 \\}] = E_{y \sim Q} [max \\{ \frac{P(y)}{Q(y)} - e^{\epsilon}, 0 \\}] \leq \delta$$

推导：差分隐私表示为常见的形式化定义如下：

$$P[M(D) \in S] \leq e^{\epsilon} P[M(D' \in S)] + \delta$$

$$P(y) \leq e^{\epsilon} Q(y) + \delta$$

$$P(y) - e^{\epsilon} Q(y) \leq \delta$$

$$max \\{P(y) - e^{\epsilon} Q(y), 0\\} \leq \delta$$

$$\int max \\{P(y) - e^{\epsilon} Q(y), 0\\} dy \leq \delta$$

$$\int Q(y) max \\{\frac{P(y)}{Q(y)} - e^{\epsilon}, 0\\} dy \leq \delta$$

$$E_{y \sim Q} [max \\{ \frac{P(y)}{Q(y)} - e^{\epsilon}, 0 \\}] \leq \delta$$

### 蒙特卡洛隐私会计

上述期望无法计算或计算较为复杂，因此引入蒙特卡洛进行近似计算，通过抽样和构造统计置信区间来证明担保隐私性。

$$\hat{\delta} = \frac{1}{s} \sum^s_{i=1} max \\{ 1 - e^{\epsilon} \frac{Q(y_i)}{P(y_i)}, 0 \\}, \quad \quad y_i \sim P$$

第一个问题：在蒙特卡洛隐私会计中，高效计算这个似然比 $\frac{P(y)}{Q(y)}$ 是关键挑战。

第二个问题：因为经验估计值 $\hat{\delta}$ 是一个随机变量，所以仅靠蒙特卡洛估计并不能直接提供形式化的隐私保证。

> 例子：假设蒙特卡洛计算出的 $\hat{\delta}$ 是1e-5，但是真实的值是 1.5e-5，因为蒙特卡洛给出的值带有统计误差，估计本身有置信区间。

解决方案：引入 Estimate-Verify-Release (EVR) 框架，（1）将统计验证器的失败概率吸收到总的 $\delta$ 预算中；（2）不使用 $\hat{\delta}$ ，而是使用统计置信上界；（3）确保即使考虑统计误差，仍然满足 $(\epsilon, \delta)-DP$ 。

在本文评估中，采用了改进的 EVR 蒙特卡洛隐私会计框架，解决第二个问题（已有工作）。因此本文主要专注于解决高效地评估平衡迭代子采样（Balanced Iteration Subsampling）的精确似然比（专注于解决第一个问题）。

## 隐私放大的最优子采样

定义设计空间：设一个训练过程由总共T次迭代，并设 $k \in (0, T)$ 为期望参与次数。将 $P_{T, k}$ 定义为布尔超立方体 $\\{0, 1\\}^T$ 上满足期望参与次数约束的所有概率分布 P 的集合：

$$E_{x \sim P} \parallel x \parallel^2_2 = \sum_{x \in \\{0, 1\\}^T} P(x) \parallel x \parallel^2_2 = k$$

将独立样本子采样方案（independent-example subsampling schemes）定义为一批次构建机制的集合，其中每个样本 i 的参与向量 $x^{(i)}$ 是从某个 P 中独立同分布（i.i.d.）抽取的。

> 由上述设计空间可知，泊松子采样和BIS都是独立样本采样方案。不同点在于，泊松子采样有方差波动，而BIS的方差为0。

