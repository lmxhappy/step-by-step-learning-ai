# 推荐算法(2025.1.31)：Meta HyperZero，用小时级反馈自动调优融合公式超参数

> 原文：https://mp.weixin.qq.com/s/k7nEXZJqXf6nJ96EkpgPzw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-04（标题日期 2025.1.31）
> 本文拆自《推荐算法paper（2025.1.31）》，同期 3 篇：[阿里 DRP 搜索行为建模空间重构](../behavior-sequence-modeling/2025-02-04-ali-drp-search-behavior.md)、[Meta HyperZero 融合公式自动调参](../multi-task-learning/2025-02-04-meta-hyperzero.md)、[Amazon 序列推荐最小最大群体公平](../loss-func/2025-02-04-amazon-minimax-group-fairness.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-04-meta-hyperzero.full.en.md)

HyperZero: A Customized End-to-End Auto-Tuning System for Recommendation with Hourly Feedback

网址：https://arxiv.org/pdf/2501.18126

公司：meta

方向：rerank融合公式

解读：融合公式（paper称之为价值模型阶段）的超参数的自动化调优。利用每小时系统反馈数据。先通过特殊的小时级信号处理非独立同分布的噪声数据，再用高斯过程和汤普森采样结合的零阶优化方法探索超参数空间，同时采用异步并行探索加快找最优解，实现高效调优。

可信度：生产

推荐等级：推荐看看
