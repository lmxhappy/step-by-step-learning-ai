# 推荐算法paper清单（2024.11.18）

> 原文：https://mp.weixin.qq.com/s/-UD2-yNlJtuYUJYGRQkagA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-18

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-18-three-papers.full.en.md)

1 KuaiFormer: Transformer-Based Retrieval at Kuaishou

网址：https://arxiv.org/pdf/2411.10057

公司：快手。DIN作者周国瑞团队的。

方向：召回+transforer

概括：将transformer用于召回。由于物料量天然对召回模型的效率有高要求，所以transformer在实时召回上没有落地。这个工作将其落地了，且做到billion物料量、多兴趣、长序列。

效果：生产。

推荐等级：值得看看。

2 InterFormer: Towards Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction

网址：https://arxiv.org/pdf/2411.09852

公司：meta

方向：rank+多模态

概括：提升异构数据在多模态模型中的效果。多模态有两个挑战，一是多种异构数据交互不够，二是每种异构数据的信息过早的聚合。这个工作尝试解决这两个挑战。感兴趣的同学看看到底如何解决的。

效果：生产

推荐等级：值得看看

3 Residual Multi-Task Learner for Applied Ranking

网址：https://arxiv.org/pdf/2411.09705

公司：shopee

方向：搜索rank+多任务

概括：在多个任务之间，中间layer的output加到其它task上。

效果：生产

推荐等级：值得看看
