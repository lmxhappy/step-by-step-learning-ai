# 推荐算法paper清单（2024.11.04）

> 原文：https://mp.weixin.qq.com/s/cunko32RxO5bvXuxTkawzQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-03

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-04-two-papers.full.en.md)

1. Beyond Utility: Evaluating LLM as Recommender

a. 网址：https://arxiv.org/pdf/2411.00331

b. 公司：华为

c. 方向：LLMs推荐。关于评估LLMs推荐模型，提出了4个的评估维度，并提出了一个多维评估框架。并且发现，LLM推荐模型在处理具有先验知识和较短输入历史的排序任务中表现出色，并且在重排序任务中表现更佳，在多个维度上超越了传统模型，但是在候选位置偏差方面存在显著问题，一些模型更频繁地生成不存在的物品。题外话：通过这篇论文，您可以全面了解LLMs的发展阶段、优缺点以及所面临的问题。

2. DivNet: Diversity-Aware Self-Correcting Sequential Recommendation Networks

a. 网址：https://arxiv.org/pdf/2411.00395

b. 公司：阿里

c. 方向：rerank+解决多样性。通过self-attention networks，实现兼顾最终item list的多样性、整体用户utility、exposure bias的rerank。题目里不叫rerank，而是collective recommendation，还真不了解这个，恕我无知，还需努力。
