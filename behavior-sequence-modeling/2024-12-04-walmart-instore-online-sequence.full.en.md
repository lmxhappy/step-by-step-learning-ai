# Recommendation papers (2024.12.4): Walmart fuses in-store behavior into sequential recommendation
> [中文版](2024-12-04-walmart-instore-online-sequence.md)

> Source: WeChat post https://mp.weixin.qq.com/s/EZNw3eYCg2p3qbzZ-6Awuw, published 2024-12-04.
> Split from the roundup "Recommendation papers (2024.12.4)"; same issue (2 papers): [Pinterest: feature interaction experience](../feature-cross/2024-12-04-pinterest-feature-interaction.full.en.md), [Walmart: online + in-store behavior sequences](../behavior-sequence-modeling/2024-12-04-walmart-instore-online-sequence.full.en.md).

Improving Sequential Recommender Systems with Online and In-store User Behavior

Link: https://arxiv.org/pdf/2412.02122

Company: Walmart

Area: behavior sequence + data fusion

Overview: Collects in-store behavior as a supplement to online behavior. Rather than adding in-store behavior directly into the online behavior sequence, it passes segments of the in-store behavior sequence through an encoder to get an item-set embedding, and then does sequence modeling on it together with the online behavior sequence.

Credibility: offline validation on production data.

Recommendation: worth reading.
