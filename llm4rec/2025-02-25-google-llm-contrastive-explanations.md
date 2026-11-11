# 推荐算法(2025.2.25)：Google LR-Recsys 用大模型生成正负对比解释增强黑盒推荐

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484189&idx=1&sn=c256db252d5aa7ce3629b8c45953a2d5&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-25
> 本文拆自《推荐算法paper速冲！四大厂最新实用科技成果》，同期 4 篇：[Meta 语义与 ID 统一表征](../sid/2025-02-25-meta-unified-semantic-id.md)、[Google LR-Recsys 对比解释](../llm4rec/2025-02-25-google-llm-contrastive-explanations.md)、[华为 SCCDR 跨域对比学习](../cross-domain-recommendation/2025-02-25-huawei-separated-contrastive-cdr.md)、[阿里 ESANS 语义感知负采样](../deep-retrieval/2025-02-25-ali-esans-negative-sampling.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-25-google-llm-contrastive-explanations.full.en.md)

当期前言：

这篇文章主要聚焦于推荐系统领域的前沿研究，涉及 LLM 推荐、跨域推荐和向量召回等方向，解读了四篇来自知名企业的论文，并给出了心得和推荐等级。

The Blessing of Reasoning: LLM-Based Contrastive Explanations in Black-Box Recommender Systems

网址：https://arxiv.org/pdf/2502.16759

公司：google

方向：LLM推荐

解读：将大语言模型（LLMs）与深度神经网络（DNNs）相结合，能够同时提升这两方面的性能。

LR-Recsys 引入了一个对比解释生成器，借助 LLMs 生成人类可读的正面解释（消费者可能喜欢某产品的原因）和负面解释（消费者可能不喜欢某产品的原因）。这些解释通过微调后的自动编码器进行嵌入，并将其输入到 DNN 中进行融合。

可信度：离线

推荐等级：值得看看
