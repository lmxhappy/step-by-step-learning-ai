# Recommendation papers (2025.1.14): Amazon uses a multimodal two-tower model for semantic retrieval in product search
> [中文版](2025-01-14-amazon-multimodal-product-retrieval.md)

> Source: WeChat post https://mp.weixin.qq.com/s/cDhNZfJwkUpI-Zl8jJieag, published 2025-01-14.
> Split from the roundup "Recommendation papers (2025.1.14)"; same issue (2 papers): [UNIST intent-interest disentanglement](../behavior-sequence-modeling/2025-01-14-academic-intent-interest-disentangle.full.en.md), [Amazon multimodal product-search retrieval](../deep-retrieval/2025-01-14-amazon-multimodal-product-retrieval.full.en.md).

Multimodal semantic retrieval for product search

Link: https://arxiv.org/pdf/2501.07365

Company: Amazon

Area: As the title says: product search + semantic retrieval + multimodal

Overview: Product images are crucial to e-commerce search interactions and are a key factor for customers during product exploration, but their impact on semantic retrieval has not been studied thoroughly. In this work, the authors build multimodal representations for products in e-commerce search, contrast them with text-only product representations, and study the impact of these representations. In terms of implementation, it is still a two-tower architecture, except that each tower is split into two sub-towers: besides the text-based BiBERT, there is also an image CLIP tower.

Credibility: Online

Recommendation: Worth reading
