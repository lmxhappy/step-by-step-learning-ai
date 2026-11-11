# 推荐算法paper清单（2024.11.05）

> 原文：https://mp.weixin.qq.com/s/WundhKOYdM93hyuwm6qtdg
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-04

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-05-three-papers.full.en.md)

1. LLM4PR: Improving Post-Ranking in Search Engine with Large Language Models

a. 网址：https://arxiv.org/pdf/2411.01178

b. 公司：快手

c. 方向：搜索+post-ranking+LLM。将LLM技术应用在搜索的post-ranking环节。

d. 可信度：offine验证。MovieLens-1M & KuaiSAR

2. Enhancing ID-based Recommendation with Large Language Models

a. 网址：https://arxiv.org/pdf/2411.02041

b. 公司：美团

c. 方向：LLM。增强LLM理解ID数据和基于ID的推荐任务的能力。通俗的讲，普通的LLM模型输入和输出都是一些文本数据，本文希望LLM输入输出的是ID，包括用户id和item id。并通过finetune的LLM模型，生成增强的item id，通过这些增强数据进一步提高模型的效果。

d. 可信度：offline验证

3. LinRec: Linear Attention Mechanism for Long-term Sequential Recommender Systems

a. 网址：https://arxiv.org/pdf/2411.01537

b. 公司：蚂蚁

c. 方向：长行为序列+attention。LinRec具有线性复杂度，同时保留了注意力机制的特性。待感兴趣的同学深入了解。

d. 可信度：offline验证。

e. 代码：https://github.com/Applied-Machine-Learning-Lab/LinRec

后记：大模型paper真多。
