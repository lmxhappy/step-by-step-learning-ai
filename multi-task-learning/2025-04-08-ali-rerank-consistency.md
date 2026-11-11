# 推荐算法(2025.4.8)：阿里重排的两个原则，收敛一致性与对抗一致性

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484424&idx=1&sn=103d2fab460bb03ccfd332c5f1c2f0a3&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-08
> 本文拆自《推荐算法(2025.4.8)：魔鬼藏在细节里》，同期 3 篇：[Meta 片段信息 CTR 预估](../feature-cross/2025-04-08-meta-segment-ctr.md)、[阿里 Masked Softmax Loss](../generative-recommendation/2025-04-08-ali-masked-softmax-loss.md)、[阿里重排一致性原则](../multi-task-learning/2025-04-08-ali-rerank-consistency.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-04-08-ali-rerank-consistency.full.en.md)

当期前言：

（1）这篇的3篇paper都值得看，虽然paper中用到的技术可能不是很新颖，但是在魔鬼藏在细节里。

**魔鬼藏在细节里，这句话值得我们在工作中不断提醒自己！**

Towards Principled Learning for Re-ranking in Recommender Systems

网址：https://arxiv.org/pdf/2504.04188

公司：阿里

方向：rerank

解读：

本文首次提出了收敛一致性和对抗一致性原则，用于指导重新排序器的学习和评估输出质量。这些原则可作为即插即用的组件，提升现有重新排序器的性能，为推荐系统的重新排序研究提供了新方向。

（1）收敛一致性：一个重新排序器（re-ranker）对初始列表进行重新排序后得到的输出列表，不能被同一重新排序器再次调整。初始列表 A 输入重新排序器，输出列表 B；将 B再次输入同一重新排序器，输出仍应为 B。

（2）对抗一致性：对抗一致性原则要求：初始排序列表的微小扰动（如相邻项目位置交换）不应改变重新排序器的输出结果。初始列表 A 和其扰动版本 B（包含相同项目但顺序略有不同），输入重新排序器后应输出相同的列表 C。

心得：

有原则，就遵循原则，没有原则就要发现原则，注意不是创造原则。

可信度：离线

推荐等级：有时间看看

