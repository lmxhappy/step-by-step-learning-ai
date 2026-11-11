# Recommendation papers (2025.1.28): ByteDance AdaF2M2: combining feature interaction with sequence modeling
> [中文版](2025-02-07-bytedance-adaf2m2.md)

> Source: WeChat post https://mp.weixin.qq.com/s/pQgDNxLJ7vBt_QUB4safyA, published 2025-02-07.
> Split from the roundup "Recommendation algorithm papers (2025.1.28)"; same issue: [ByteDance Long-Term Interest Clock](../behavior-sequence-modeling/2025-02-07-bytedance-long-term-interest-clock.full.en.md), [ByteDance AdaF2M2](../feature-cross/2025-02-07-bytedance-adaf2m2.full.en.md), [Tencent PatchRec](../llm4rec/2025-02-07-tencent-patchrec.full.en.md).

AdaF2M2: Towards Feature-adaptive and Multi-level Model Mixture for Sequential Recommendation

Link: https://arxiv.org/pdf/2501.15816

Company: ByteDance

Area: features

Overview: Another innovation from ByteDance: combining feature interaction from feature engineering with sequence modeling. Specifically:

(1) Feature masking: when the model processes the user behavior sequence, some feature dimensions are randomly masked, forcing the model to learn more robust feature representations and improving generalization when data is incomplete or noisy. This mechanism is especially suited to the sparsity and uncertainty in user behavior sequences.

(2) State-aware adapter: dynamically adjusts the feature fusion strategy based on the current user state (interest shifts, context, etc.). The adapter captures subtle changes in user state and optimizes the feature representation accordingly, improving accuracy and personalization. If I read it right, it looks like the EPNet in PEPNet.

Credibility: production

Recommendation: worth reading
