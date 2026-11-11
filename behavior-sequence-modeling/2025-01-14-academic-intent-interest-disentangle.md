# 推荐算法(2025.1.14)：UNIST解耦意图与兴趣做序列推荐

> 原文：https://mp.weixin.qq.com/s/cDhNZfJwkUpI-Zl8jJieag
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-14
> 本文拆自《推荐算法paper（2025.1.14）》，同期 2 篇：[UNIST 意图-兴趣解耦序列推荐](../behavior-sequence-modeling/2025-01-14-academic-intent-interest-disentangle.md)、[亚马逊多模态商品搜索召回](../deep-retrieval/2025-01-14-amazon-multimodal-product-retrieval.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-14-academic-intent-interest-disentangle.full.en.md)

Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation

网址：https://arxiv.org/pdf/2501.07096

公司：大学UNIST(韩国的一所大学)

方向：行为序列+意图建模

解读：将用户行为分解为意图（即用户的动态动机）和兴趣（即用户的稳定偏好），以全面理解用户行为。因果交叉注意力机制用于识别跨交互的一致兴趣，而剩余行为通过相似性调整损失对其时间动态进行建模，从而被建模为意图。

此外，在不预先定义意图类别数量的情况下，重要性加权注意力机制会考虑每个交互中意图的重要性来捕捉特定用户的类别意图。进一步地，引入了物品感知对比学习，它使在同一交互中出现的意图对齐，并使意图与相应意图所产生的物品组合对齐。

可信度：离线

推荐等级：值得看看
