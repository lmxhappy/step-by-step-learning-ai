# 推荐算法(2025.1.31)：Amazon用DRO提升序列推荐的最小最大群体公平性

> 原文：https://mp.weixin.qq.com/s/k7nEXZJqXf6nJ96EkpgPzw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-04（标题日期 2025.1.31）
> 本文拆自《推荐算法paper（2025.1.31）》，同期 3 篇：[阿里 DRP 搜索行为建模空间重构](../behavior-sequence-modeling/2025-02-04-ali-drp-search-behavior.md)、[Meta HyperZero 融合公式自动调参](../multi-task-learning/2025-02-04-meta-hyperzero.md)、[Amazon 序列推荐最小最大群体公平](../loss-func/2025-02-04-amazon-minimax-group-fairness.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-04-amazon-minimax-group-fairness.full.en.md)

Improving Minimax Group Fairness in Sequential Recommendation

网址：https://arxiv.org/pdf/2501.18117

公司：amazon

方向：行为序列+fairness

解读：这篇论文旨在解决序列推荐中特定用户群体推荐效果不佳的问题，聚焦最小最大群体公平性。作者探索了 DRO（一种优化方法，旨在处理优化问题中的不确定性）相关的三种方法，通过实验对比发现，DRO 方法总体优于标准训练，其中 CVaR DRO 效果最佳，且对交叉用户群体的场景有更好的适应性。

CVaR：是指在一定的置信水平下，当损失超过 VaR 时的平均损失值。它关注的是超过 VaR 的那部分损失的平均情况，即回答 "当损失超过 VaR 时，平均损失是多少" 的问题。例如，在上述例子中，如果 CVaR 为 80 万元，意味着当损失超过 50 万元时，平均损失将达到 80 万元。

可信度：离线

推荐等级：有时间看看看看
