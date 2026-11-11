# 推荐算法(2024.12.6)：腾讯用 LLM 文本与行为序列 MoE 融合缓解物料冷启动

> 原文：https://mp.weixin.qq.com/s/v7NZ4K9t520KMFIZc8n3PA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-06
> 本文拆自《推荐算法paper（2024.12.6）》，同期 2 篇：[Salesforce 序列与图对齐](../behavior-sequence-modeling/2024-12-06-salesforce-graph-sequence-alignment.md)、[腾讯 LLM 赋能序列推荐](../llm4rec/2024-12-06-tencent-llm-sequence-disentangle.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-06-tencent-llm-sequence-disentangle.full.en.md)

Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models

网址：https://arxiv.org/pdf/2412.04107

公司：腾讯

方向：行为序列+LLM+冷启动

解读：分别训练行为序列model和文本encoder的LM模型，然后将两者合在一起构建了3个expert，以MOE的方式一起训练。从而希望让文本信息能够缓解物料冷启动问题。

可信度：离线

推荐等级：有时间看看
