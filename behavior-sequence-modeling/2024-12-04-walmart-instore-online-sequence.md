# 推荐算法(2024.12.4)：沃尔玛融合线下门店行为的序列推荐

> 原文：https://mp.weixin.qq.com/s/EZNw3eYCg2p3qbzZ-6Awuw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-04
> 本文拆自《推荐算法paper（2024.12.4）》，同期 2 篇：[Pinterest 特征交叉经验](../feature-cross/2024-12-04-pinterest-feature-interaction.md)、[沃尔玛线上线下行为序列](../behavior-sequence-modeling/2024-12-04-walmart-instore-online-sequence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-04-walmart-instore-online-sequence.full.en.md)

Improving Sequential Recommender Systems with Online and In-store User Behavior

网址：https://arxiv.org/pdf/2412.02122

公司：沃尔玛

方向：行为序列+数据融合

解读：将店里的行为收集起来，作为online行为的补充。不是将店里行为直接加入到online行为序列里，而是将店里的行为序列片段经过一个encoder获得一个item set embed，再跟online的行为序列，一起再做序列建模。

可信度：生产数据离线验证。

推荐等级：推荐看看。
