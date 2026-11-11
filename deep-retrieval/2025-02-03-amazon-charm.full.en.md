# Recommendation papers (2025.2.3): Amazon CHARM: hierarchical multi-field representations for two-stage e-commerce retrieval
> [中文版](2025-02-03-amazon-charm.md)

> Source: WeChat post https://mp.weixin.qq.com/s/qvI9Ac5AJ96QwL3DoSPP5Q, published 2025-02-03.
> Split from the roundup "Recommendation algorithm papers (2025.2.3)"; same issue: [Amazon CHARM multi-field retrieval](../deep-retrieval/2025-02-03-amazon-charm.full.en.md), [Google collaborative diffusion](../representation-quality/2025-02-03-google-collaborative-diffusion.full.en.md).

Hierarchical Multi-field Representations for Two-Stage E-commerce Retrieval

Link: https://arxiv.org/pdf/2501.18707

Company: Amazon

Area: search + deep retrieval

Overview: The paper proposes the Cascading Hierarchical Attention Retrieval Model (CHARM) for e-commerce product retrieval, which captures the interdependencies between product fields within a specific hierarchy. The core of the method is a block-triangular attention mechanism that lets the tokens of a product field attend to themselves and to the tokens of preceding fields, combined with the product field hierarchy to produce field-level representations. These representations are then aggregated for an initial screening match, followed by re-evaluation through precise comparison against the individual field representations, giving an efficient retrieval pipeline.

Credibility: offline

Recommendation: read if you have time
