# 推荐算法paper清单（2024.10.30）

> 原文：https://mp.weixin.qq.com/s/TGlyp1svAwicQpRdMqJOfw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-10-30

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-10-30-four-papers.full.en.md)

1. SimRec: Mitigating the Cold-Start Problem in Sequential Recommendation by Integrating Item Similarity

a. 网址：https://arxiv.org/pdf/2410.22136

b. 公司：Amazon

c. 方向：行为序列建模+物料冷启+对比学习。通过定义了一个结合结合对比学习技术的新Loss，可以实现对新物料的推荐效果。看了一下，方法很简单，详情还待深入研究。

d. 推荐等级：强烈建议

2. Deep Group Interest Network on Full Lifelong User Behaviors for CTR Prediction

a. 网址：https://arxiv.org/pdf/2311.10764

b. 公司：美团

c. 方向：长行为序列。考虑用户的所有行为，而且建模多种互动行为。而且通过group embed增强模型效果——这是最近很流行的enhance手段。除此之外，还给出了一个证明必要性的数据分析，这个更有价值，对于我们发现问题，设计解决问题的方法，很有启发意义。

d. 推荐等级：强烈推荐

3. Enhancing CTR prediction in Recommendation Domain with Search Query Representation

a. 网址：https://arxiv.org/pdf/2410.21487

b. 公司：华为

c. 方向：跨域+对比学习。搜索场景的数据，帮助提升推荐场景的效果。

d. 推荐等级：值得看

4. Modeling Temporal Positive and Negative Excitation for Sequential Recommendation

a. 网址：https://arxiv.org/pdf/2410.22013

b. 公司：澳大利亚一个不了解的公司

c. 方向：行为序列建模。一般的行为序列建模是为了挖掘用户的动态兴趣，不考虑静态兴趣，只考虑正向行为的item。本paper同时考虑动态兴趣和静态兴趣，同时也考虑负向行为的item。

d. 推荐等级：有时间看看。不要期待有落地效果，只是觉得挺有趣的，有时间的话，开阔一些眼界也好。
