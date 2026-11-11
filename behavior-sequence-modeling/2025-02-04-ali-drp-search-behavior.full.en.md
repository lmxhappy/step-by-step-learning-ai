# Recommendation papers (2025.1.31): Alibaba DRP: decoupling relevance from preference to rebuild the behavior modeling space in e-commerce search
> [中文版](2025-02-04-ali-drp-search-behavior.md)

> Source: WeChat post https://mp.weixin.qq.com/s/k7nEXZJqXf6nJ96EkpgPzw, published 2025-02-04.
> Split from the roundup "Recommendation algorithm papers (2025.1.31)"; same issue: [Alibaba DRP search behavior modeling](../behavior-sequence-modeling/2025-02-04-ali-drp-search-behavior.full.en.md), [Meta HyperZero fusion-formula auto-tuning](../multi-task-learning/2025-02-04-meta-hyperzero.full.en.md), [Amazon minimax group fairness](../loss-func/2025-02-04-amazon-minimax-group-fairness.full.en.md).

Behavior Modeling Space Reconstruction for E-Commerce Search

Link: https://arxiv.org/pdf/2501.18216

Company: Alibaba

Area: search

Overview: Proposes the DRP framework, which reconstructs the behavior modeling space with two core components. The first is preference editing: by finding a suitable orthogonal low-rank matrix, it removes the influence of relevance from the preference prediction, obtaining a purer user preference and reconstructing a fine-grained modeling space. The second is dual-level adaptive fusion, consisting of global adaptive fusion and local adaptive fusion. Global adaptive fusion introduces learnable parameters to distinguish the patterns of different relevance-preference states; local adaptive fusion further fits the differences between behavior signals within similar states, so the model captures user behavior patterns in finer detail.

Decoupling has been a method that works in recent years; it works because it better balances the influence of multiple factors.

Credibility: online

Recommendation: worth reading
