# Recommendation papers (2025.1.28): Tencent PatchRec: patch compression lets an LLM handle long behavior sequences
> [中文版](2025-02-07-tencent-patchrec.md)

> Source: WeChat post https://mp.weixin.qq.com/s/pQgDNxLJ7vBt_QUB4safyA, published 2025-02-07.
> Split from the roundup "Recommendation algorithm papers (2025.1.28)"; same issue: [ByteDance Long-Term Interest Clock](../behavior-sequence-modeling/2025-02-07-bytedance-long-term-interest-clock.full.en.md), [ByteDance AdaF2M2](../feature-cross/2025-02-07-bytedance-adaf2m2.full.en.md), [Tencent PatchRec](../llm4rec/2025-02-07-tencent-patchrec.full.en.md).

PatchRec: Patching and Compression for Efficient LLM-Based Sequential Recommendations

Link: https://arxiv.org/pdf/2501.15087

Company: Tencent

Area: LLM + long sequences

Overview: Tencent's innovative paper on LLMs + long sequences. The main idea is to use the strong capabilities of an LLM to solve long-sequence modeling.

Through multi-granularity compression and two-stage training:

(1) Hierarchical compression: a multi-granularity patch compression mechanism converts each item in the raw behavior sequence into patches of different granularity, and hierarchical compression reduces the number of input tokens. Patches can be further compressed hierarchically for a more efficient sequence representation. (2) Two-stage training: patch pre-training and patch fine-tuning. Patch pre-training learns from both raw and compressed samples at the same time, so the model builds the correspondence between item patches and the original text tokens; patch fine-tuning proposes two strategies with different compression granularity, PFT-I and PFT-S, so the model adapts to a multi-granularity, time-aware compression format and captures the temporal decay of user interest. PFT-S, so the model adapts to a multi-granularity, time-aware compression format and captures the temporal decay of user interest.

Credibility: offline

Recommendation: worth reading
