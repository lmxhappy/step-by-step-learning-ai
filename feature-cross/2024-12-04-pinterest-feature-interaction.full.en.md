# Recommendation papers (2024.12.4): Pinterest's experience with feature interactions under industry constraints
> [中文版](2024-12-04-pinterest-feature-interaction.md)

> Source: WeChat post https://mp.weixin.qq.com/s/EZNw3eYCg2p3qbzZ-6Awuw, published 2024-12-04.
> Split from the roundup "Recommendation papers (2024.12.4)"; same issue (2 papers): [Pinterest: feature interaction experience](../feature-cross/2024-12-04-pinterest-feature-interaction.full.en.md), [Walmart: online + in-store behavior sequences](../behavior-sequence-modeling/2024-12-04-walmart-instore-online-sequence.full.en.md).

Improving feature interactions at Pinterest under industry constraints

Link: https://arxiv.org/pdf/2412.01985

Company: Pinterest

Area: feature interaction

Overview: Describes Pinterest's experience with feature interactions, covering both feature processing and feature crossing. (1) Feature processing: normalize dense features; for sparse features, choose a suitable embedding size and reduce the dimension of the large ones; apply L2 norm to the embeddings of both sparse features and embedding features. (2) Feature crossing: DCNv2. It also discusses other things, such as the challenges they faced and how they solved them. This is a thoroughly industrial paper, especially worth recommending.

Credibility: production

Recommendation: strongly recommended.
