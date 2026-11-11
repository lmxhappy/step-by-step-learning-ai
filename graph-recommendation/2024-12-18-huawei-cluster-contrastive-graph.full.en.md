# Recommendation papers (2024.12.18): Huawei uses cluster-guided contrastive learning for class-imbalanced graph classification
> [中文版](2024-12-18-huawei-cluster-contrastive-graph.md)

> Source: WeChat post https://mp.weixin.qq.com/s/-m8X1ILtc_5878LzriS_JQ, published 2024-12-18.
> Split from the roundup "Recommendation papers (2024.12.18)"; same issue: [Alipay LLM relevance modeling](../llm4rec/2024-12-18-ant-llm-relevance-robust.full.en.md), [Huawei class-imbalanced graph classification](../graph-recommendation/2024-12-18-huawei-cluster-contrastive-graph.full.en.md).

Cluster-guided Contrastive Class-imbalanced Graph Classification

Link: https://arxiv.org/pdf/2412.12984

Company: Huawei

Area: GNN. This paper has nothing to do with recommendation algorithms, but GNNs can be used for recommendation.

Overview: The method brings the idea of clustering into contrastive learning to improve class-imbalanced graph classification. (1) Cluster the graphs of each majority class into multiple subclasses so that they are similar in size to the minority classes, which alleviates class imbalance; this way, the semantic substructures within the majority classes can be fully explored. (2) Use the Mixup technique to synthesize new samples and enrich the semantic information of each subclass, and use supervised contrastive learning to learn effective graph representations hierarchically; this effectively alleviates over-attention to the minority classes.

Credibility: offline

Recommendation: worth reading
