# 推荐算法(2025.2.3)：Amazon CHARM，分层多字段表示做两阶段电商检索

> 原文：https://mp.weixin.qq.com/s/qvI9Ac5AJ96QwL3DoSPP5Q
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-03
> 本文拆自《推荐算法paper（2025.2.3）》，同期 2 篇：[Amazon CHARM 分层多字段检索](../deep-retrieval/2025-02-03-amazon-charm.md)、[Google 协同扩散推荐](../representation-quality/2025-02-03-google-collaborative-diffusion.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-03-amazon-charm.full.en.md)

Hierarchical Multi-field Representations for Two-Stage E-commerce Retrieval

网址：https://arxiv.org/pdf/2501.18707

公司：amazon

方向：搜索+深度召回

解读：该论文提出级联分层注意力检索模型（CHARM）用于电商产品检索，能够捕捉特定层次结构中产品字段之间的相互依赖关系。方法核心是利用块三角注意力机制，让产品字段词元关注自身及之前字段词元，结合产品字段层次结构生成字段级表示。再聚合这些表示用于初步筛选匹配，之后通过与单个字段表示精确比对重新评估，实现高效检索流程 。

可信度：离线

推荐等级：有时间看看
