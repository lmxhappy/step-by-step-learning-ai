# Recommendation papers (2024.11.25): Huawei MIRRN retrieves and refines long-term user interest at multiple granularities
> [中文版](2024-11-25-huawei-mirrn-multi-granularity-interest.md)

> Source: WeChat post https://mp.weixin.qq.com/s/Tui8ZesqJjpCHRf2hUCRig, published 2024-11-25.
> Split from the roundup "Recommendation papers (2024.11.25)"; same issue (2 papers): [Huawei MIRRN multi-granularity interest retrieval and refinement](../behavior-sequence-modeling/2024-11-25-huawei-mirrn-multi-granularity-interest.full.en.md), [Huawei LIBER LLM-based lifelong behavior modeling](../behavior-sequence-modeling/2024-11-25-huawei-liber-llm-lifelong.full.en.md).

Multi-granularity Interest Retrieval and Refinement Network for Long-Term User Behavior Modeling in CTR Prediction

Link: https://arxiv.org/pdf/2411.15005

Company: Huawei

Area: behavior sequence modeling

Overview: Uses 3 modeling methods (e.g. a SIM-style method, a GRU/sum-pooling method, and clustering) to obtain the user's interest sub-sequences at different granularities. A novel multi-head Fourier transformer processes the 3 sub-sequences, producing 3 kinds of interest: target-aware, local-aware and global-aware. Finally, multi-head target attention fuses these 3 representations into the user's interest representation. Notably, a clustering method called Density Peak Clustering (DPC) lets it handle sequences several thousand long.

Credibility: production

Recommendation: worth a look
