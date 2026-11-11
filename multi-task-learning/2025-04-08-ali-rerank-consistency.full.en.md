# Recommendation papers (2025.4.8): two principles for Alibaba re-ranking, convergence consistency and adversarial consistency
> [中文版](2025-04-08-ali-rerank-consistency.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484424&idx=1&sn=103d2fab460bb03ccfd332c5f1c2f0a3&scene=21#wechat_redirect, published 2025-04-08.
> Split from the roundup "Recommendation papers (2025.4.8): the devil is in the details"; same issue: [Meta segment-information CTR prediction](../feature-cross/2025-04-08-meta-segment-ctr.full.en.md), [Alibaba Masked Softmax Loss](../generative-recommendation/2025-04-08-ali-masked-softmax-loss.full.en.md), [Alibaba re-ranking consistency principles](../multi-task-learning/2025-04-08-ali-rerank-consistency.full.en.md).

Intro of this issue:

(1) All 3 papers in this issue are worth reading. The techniques they use may not be very new, but the devil is in the details.

**The devil is in the details: something worth reminding ourselves of again and again at work!**

Towards Principled Learning for Re-ranking in Recommender Systems

Link: https://arxiv.org/pdf/2504.04188

Company: Alibaba

Area: rerank

Overview:

This paper is the first to propose the convergence consistency and adversarial consistency principles, used to guide re-ranker learning and to evaluate output quality. The principles work as plug-and-play components that improve existing re-rankers, and they open a new direction for re-ranking research in recommender systems.

(1) Convergence consistency: the output list a re-ranker produces from an initial list cannot be further adjusted by the same re-ranker. Feed initial list A into the re-ranker and get output list B; feed B into the same re-ranker again and the output should still be B.

(2) Adversarial consistency: the adversarial consistency principle requires that a small perturbation of the initial ranked list (such as swapping adjacent items) should not change the re-ranker's output. Initial list A and its perturbed version B (same items, slightly different order) should both produce the same output list C from the re-ranker.

Takeaways:

When there are principles, follow them; when there are none, discover them. Note: discover, not invent.

Credibility: offline

Recommendation: read if you have time
