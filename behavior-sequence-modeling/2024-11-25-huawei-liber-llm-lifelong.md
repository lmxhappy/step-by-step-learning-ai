# 推荐算法(2024.11.25)：华为LIBER用大模型分区总结终身用户行为

> 原文：https://mp.weixin.qq.com/s/Tui8ZesqJjpCHRf2hUCRig
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-25
> 本文拆自《推荐算法paper（2024.11.25）》，同期 2 篇：[华为 MIRRN 多粒度兴趣检索与精炼](../behavior-sequence-modeling/2024-11-25-huawei-mirrn-multi-granularity-interest.md)、[华为 LIBER 大模型终身行为建模](../behavior-sequence-modeling/2024-11-25-huawei-liber-llm-lifelong.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-25-huawei-liber-llm-lifelong.full.en.md)

LIBER: Lifelong User Behavior Modeling Based on Large Language Models

网址：https://arxiv.org/pdf/2411.14713

公司：华为

方向：长行为序列建模

解读：将用户行为序列cache下来，当缓存长度达到一个阈值就作为一个分区，就对该分区用大预言模型，进行分析，获得该分区的摘要，以及对该分区以及以往分区的关系进行分析。看了paper给出的case，感觉大预言模型不仅能作为人类的助理，还可以作为推荐模型的“助理”，是helpful的。如下图，蓝色框是发给大模型的问题，红色框是大模型的回答。看着回答是有增益的。

可信度：生产

推荐等级：值得看看

