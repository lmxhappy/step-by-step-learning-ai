# 推荐算法(2024.12.24)：Snap用图聚类做特征哈希、压缩Embedding表

> 原文：https://mp.weixin.qq.com/s/8UzNMNhhiqc8HL3SYfQ3GQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-24
> 本文拆自《推荐算法paper（2024.12.24）》，同期 4 篇：[华为多场景推荐基准 Scenario-Wise Rec](../cross-domain-recommendation/2024-12-24-huawei-scenario-wise-rec-benchmark.md)、[微软 RSLLM](../llm4rec/2024-12-24-microsoft-rsllm.md)、[Snap GraphHash 压缩 Embedding 表](../ml-infra/2024-12-24-snap-graphhash.md)、[华为 SyNeg LLM 合成困难负样本](../deep-retrieval/2024-12-24-huawei-syneg-llm-hard-negatives.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-24-snap-graphhash.full.en.md)

GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems

网址：https://arxiv.org/pdf/2412.17245

方向：特征值hash

公司：snap

解读：提出了一种基于图的方法，实现了一种特征值的hash办法，从而实现了对embedding table的压缩优化。基于图的协同信号，利用用户-物品交互图上基于模块度的二分图聚类来缩减嵌入表尺寸。

可信度：离线

推荐等级：有时间看看
