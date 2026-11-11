# 推荐算法(2024.11.22)：Snap HARec在双曲空间对齐图与LLM，兼顾探索与利用

> 原文：https://mp.weixin.qq.com/s/Qeio7ycHiLH-PXU7jCUSTw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-22
> 本文拆自《推荐算法paper（2024.11.22）》，同期 2 篇：[Snap HARec 双曲图-LLM 对齐](../long-tail/2024-11-22-snap-harec-hyperbolic.md)、[Meta 协同集成 CTR 框架](../feature-cross/2024-11-22-meta-collaborative-ensemble-ctr.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-22-snap-harec-hyperbolic.full.en.md)

HARec: Hyperbolic Graph-LLM Alignment for Exploration and Exploitation in Recommender Systems

网址：https://arxiv.org/pdf/2411.13865

公司：snap

方向：长尾+LLM

解读：超球面空间是弯曲的、非欧几里得几何，特点是远离原点的距离增长呈 指数形式，类比为树形结构中，层次越深，节点的数量指数增加。它适合层次化关系的建模，用户可能对某一大类物品感兴趣（如“电子产品”），也可能对更具体的子类物品感兴趣（如“手机”），超球面几何可以通过嵌入将这种分层兴趣结构建模得更清晰。

可信度：生产

推荐等级：值得看看
