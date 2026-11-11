# 推荐算法(2024.12.24)：微软把推荐系统当作大模型里的一种新语言（RSLLM）

> 原文：https://mp.weixin.qq.com/s/8UzNMNhhiqc8HL3SYfQ3GQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-24
> 本文拆自《推荐算法paper（2024.12.24）》，同期 4 篇：[华为多场景推荐基准 Scenario-Wise Rec](../cross-domain-recommendation/2024-12-24-huawei-scenario-wise-rec-benchmark.md)、[微软 RSLLM](../llm4rec/2024-12-24-microsoft-rsllm.md)、[Snap GraphHash 压缩 Embedding 表](../ml-infra/2024-12-24-snap-graphhash.md)、[华为 SyNeg LLM 合成困难负样本](../deep-retrieval/2024-12-24-huawei-syneg-llm-hard-negatives.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-24-microsoft-rsllm.full.en.md)

Towards a Unified Paradigm: Integrating Recommendation Systems as a New Language in Large Models

网址：https://arxiv.org/pdf/2412.16933

公司：微软

方向：LLM推荐

解读：本文探索了将大型语言模型 (LLMs) 用于序列推荐的方法，该方法旨在根据用户过去的行为预测用户未来的互动。 引入了一个新概念，“将推荐系统整合为大型模型中的一种新语言” (RSLLM)，它结合了传统推荐器和 LLMs 的优势。RSLLM 采用了一种独特的提示方法，将传统推荐模型中基于 ID的物品嵌入与文本物品特征融合在一起。它将用户的序列行为视为一种独特的语言，并利用一个投影层将 ID 嵌入与 LLM 的输入空间进行对齐。

可信度：离线

推荐等级：推荐看看
