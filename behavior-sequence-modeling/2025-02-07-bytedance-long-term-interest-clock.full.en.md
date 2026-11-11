# Recommendation papers (2025.1.28): ByteDance Long-Term Interest Clock: fine-grained time perception in long-sequence modeling
> [中文版](2025-02-07-bytedance-long-term-interest-clock.md)

> Source: WeChat post https://mp.weixin.qq.com/s/pQgDNxLJ7vBt_QUB4safyA, published 2025-02-07.
> Split from the roundup "Recommendation algorithm papers (2025.1.28)"; same issue: [ByteDance Long-Term Interest Clock](../behavior-sequence-modeling/2025-02-07-bytedance-long-term-interest-clock.full.en.md), [ByteDance AdaF2M2](../feature-cross/2025-02-07-bytedance-adaf2m2.full.en.md), [Tencent PatchRec](../llm4rec/2025-02-07-tencent-patchrec.full.en.md).

Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System

Link: https://arxiv.org/pdf/2501.15817

Company: ByteDance

Area: long behavior sequences + time modeling

Overview: Achieves fine-grained time perception when modeling long sequences. It borrows SIM's two steps, GSU and ESU, and adds time to both. As a reminder: GSU searches a short sub-sequence out of the long sequence; ESU does interest modeling on that short sequence. In this paper: (1) GSU: on top of SIM's GSU logic, a time-similarity term is added; (2) ESU: on top of SIM's GSU logic, the embeddings of the sequence items are concatenated with the time gap features Δ, √Δ, Δ2, log(Δ + 1)), which increases the influence of time on the interest embedding.

This paper is the follow-up to Interest clock: Time perception in real-time streaming recommendation system (https://arxiv.org/pdf/2404.19357).

Users' temporal regularity is obvious, and more and more work models it; see my write-up of another paper: [推荐算法paper](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484070&idx=1&sn=9a53aa7c49f700d68e2c3a8011bf4aeb&scene=21#wechat_redirect)

This paper is a micro-innovation: it borrows the SIM idea on top of the short-sequence version. Even so, it is pretty good.

Recommendation: production, Douyin Music

Credibility: worth reading
