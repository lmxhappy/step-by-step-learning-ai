# 推荐算法(2024.12.6)：Salesforce 联合训练行为序列与 GNN，对齐统一嵌入空间

> 原文：https://mp.weixin.qq.com/s/v7NZ4K9t520KMFIZc8n3PA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-06
> 本文拆自《推荐算法paper（2024.12.6）》，同期 2 篇：[Salesforce 序列与图对齐](../behavior-sequence-modeling/2024-12-06-salesforce-graph-sequence-alignment.md)、[腾讯 LLM 赋能序列推荐](../llm4rec/2024-12-06-tencent-llm-sequence-disentangle.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-06-salesforce-graph-sequence-alignment.full.en.md)

Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems

网址：https://arxiv.org/pdf/2412.04276

公司：salesforce

方向：行为序列+GNN

解读：之前两个是独立的。本paper希望将两者联合到一起。同时训练，共享一个统一的嵌入空间，要求两者Alignment和Uniformity，为此分别设计了相应的Loss。算是行为序列开阔的一个好方向吧。

可信度：离线

推荐等级：推荐看看
