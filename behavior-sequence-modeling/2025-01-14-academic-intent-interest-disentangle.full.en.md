# Recommendation papers (2025.1.14): UNIST disentangles intent and interest for sequential recommendation
> [中文版](2025-01-14-academic-intent-interest-disentangle.md)

> Source: WeChat post https://mp.weixin.qq.com/s/cDhNZfJwkUpI-Zl8jJieag, published 2025-01-14.
> Split from the roundup "Recommendation papers (2025.1.14)"; same issue (2 papers): [UNIST intent-interest disentanglement](../behavior-sequence-modeling/2025-01-14-academic-intent-interest-disentangle.full.en.md), [Amazon multimodal product-search retrieval](../deep-retrieval/2025-01-14-amazon-multimodal-product-retrieval.full.en.md).

Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation

Link: https://arxiv.org/pdf/2501.07096

Company: UNIST (a university in South Korea)

Area: Behavior sequences + intent modeling

Overview: Decomposes user behavior into intent (the user's dynamic motivation) and interest (the user's stable preference) to understand user behavior comprehensively. A causal cross-attention mechanism identifies the consistent interest across interactions, while the remaining behavior, whose temporal dynamics are modeled through a similarity adjustment loss, is modeled as intent.

In addition, without predefining the number of intent categories, an importance-weighted attention mechanism captures user-specific categorical intent by considering the importance of the intent in each interaction. Further, item-aware contrastive learning is introduced, which aligns the intents that appear in the same interaction, and aligns each intent with the combination of items produced by that intent.

Credibility: Offline

Recommendation: Worth reading
