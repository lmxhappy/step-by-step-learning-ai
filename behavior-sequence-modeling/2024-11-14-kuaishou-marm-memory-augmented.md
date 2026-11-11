# 推荐算法(2024.11.14)：快手MARM用缓存换计算，把序列建模扩到多层

> 原文：https://mp.weixin.qq.com/s/lQUycdEIF8DF7buOafw5wA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-14
> 本文拆自《推荐算法paper（2024.11.14）》，同期 2 篇：[快手 MARM 记忆增强多层序列建模](../behavior-sequence-modeling/2024-11-14-kuaishou-marm-memory-augmented.md)、[Amazon 语言模型先验解决冷启动](../long-tail/2024-11-14-amazon-lm-prior-cold-start.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-14-kuaishou-marm-memory-augmented.full.en.md)

MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity

网址：https://arxiv.org/pdf/2411.09425

公司：快手

方向：行为序列建模

解读：提出了一个空间换时间的方案——通过缓存部分复杂模块的计算结果，将单层注意力的序列兴趣建模模块扩展到多层。

效果：在线验证。国际业务上，每个用户2.079%的播放时长增长

推荐等级：值得看看
