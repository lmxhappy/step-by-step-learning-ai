# 推荐算法(2024.12.18)：美团用两阶段对齐和行为语义分词让LLM做推荐

> 原文：https://mp.weixin.qq.com/s/a1C-yVkShcnPD5DqaViEpw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-18
> 本文拆自《推荐算法paper（2024.12.18）》，同期 2 篇：[快手 CRM 可控召回](../deep-retrieval/2024-12-18-kuaishou-crm-controllable-retrieval.md)、[美团 LLM 推荐语义对齐](../llm4rec/2024-12-18-meituan-semantic-convergence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-18-meituan-semantic-convergence.full.en.md)

Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization

网址：https://arxiv.org/pdf/2412.13771

公司：美团

方向：LLM做推荐

解读：训练一个LLM推荐系统。将推荐系统中稀疏的物品 ID 表示转换为 LLM 可以理解的稠密语义表示，从而弥合两者之间的差距。

可信度：离线

推荐等级：有时间看看
