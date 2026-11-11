# 推荐算法(2025.1.14)：亚马逊用多模态双塔做商品搜索语义召回

> 原文：https://mp.weixin.qq.com/s/cDhNZfJwkUpI-Zl8jJieag
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-14
> 本文拆自《推荐算法paper（2025.1.14）》，同期 2 篇：[UNIST 意图-兴趣解耦序列推荐](../behavior-sequence-modeling/2025-01-14-academic-intent-interest-disentangle.md)、[亚马逊多模态商品搜索召回](../deep-retrieval/2025-01-14-amazon-multimodal-product-retrieval.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-14-amazon-multimodal-product-retrieval.full.en.md)

Multimodal semantic retrieval for product search

网址：https://arxiv.org/pdf/2501.07365

公司：amazon

方向：如题——product search+semantic retrieval+Multimodal

解读：产品图像对于电子商务搜索交互至关重要，是客户在产品探索过程中的一个关键因素。但其对语义检索的影响尚未得到充分研究。在这项研究中，为电子商务搜索中的产品构建了多模态表示，与产品的纯文本表示形成对比，并研究了这种表示的影响。实现上，还是通过双塔的架构，只是在塔里分成两个子塔，除了文本相关的Bibert，还有image的CLIP塔。

可信度：在线

推荐等级：值得看看
