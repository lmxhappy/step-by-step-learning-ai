# 推荐算法(2024.11.14)：Amazon用语言模型先验缓解物品冷启动

> 原文：https://mp.weixin.qq.com/s/lQUycdEIF8DF7buOafw5wA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-14
> 本文拆自《推荐算法paper（2024.11.14）》，同期 2 篇：[快手 MARM 记忆增强多层序列建模](../behavior-sequence-modeling/2024-11-14-kuaishou-marm-memory-augmented.md)、[Amazon 语言模型先验解决冷启动](../long-tail/2024-11-14-amazon-lm-prior-cold-start.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-14-amazon-lm-prior-cold-start.full.en.md)

Language-Model Prior Overcomes Cold-Start Items

网址：https://arxiv.org/pdf/2411.09065

公司：amazon

方向：冷启动+LLM

解读：利用语言模型（LM）来估计物品之间的相似性，并将这些相似性作为先验，提升主模型的效果。

效果：offline。

推荐等级：值得看看。开源代码
