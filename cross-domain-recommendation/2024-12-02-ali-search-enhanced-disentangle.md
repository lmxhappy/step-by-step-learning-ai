# 推荐算法(2024.12.2)：阿里用反事实学习解耦搜索特征，增强推荐

> 原文：https://mp.weixin.qq.com/s/LmvdSrwjm5jI9oVJSZcCVQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-02
> 本文拆自《推荐算法paper（2024.12.2）》，同期 2 篇：[阿里搜索增强推荐反事实解耦](../cross-domain-recommendation/2024-12-02-ali-search-enhanced-disentangle.md)、[Meta 统一生成式召回与稠密召回](../generative-recommendation/2024-12-02-meta-unify-generative-dense-retrieval.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-02-ali-search-enhanced-disentangle.full.en.md)

Counterfactual Learning-Driven Representation Disentanglement for Search-Enhanced Recommendation

https://arxiv.org/pdf/2411.18631

阿里

方向：搜索增强推荐

解读：就是从搜索场景下的item的特征，通过反事实学习，分成与query无关，与query有关等两类。只将前者用于推荐模型，提升模型效果。因为后者对推荐模型有负向作用。

可信度：生产

推荐等级：值得看看
