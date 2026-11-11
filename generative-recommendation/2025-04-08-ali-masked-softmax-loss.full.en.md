# Recommendation papers (2025.4.8): Alibaba Masked Softmax Loss, stop wrongly penalizing fictitious items in generative recommendation
> [中文版](2025-04-08-ali-masked-softmax-loss.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484424&idx=1&sn=103d2fab460bb03ccfd332c5f1c2f0a3&scene=21#wechat_redirect, published 2025-04-08.
> Split from the roundup "Recommendation papers (2025.4.8): the devil is in the details"; same issue: [Meta segment-information CTR prediction](../feature-cross/2025-04-08-meta-segment-ctr.full.en.md), [Alibaba Masked Softmax Loss](../generative-recommendation/2025-04-08-ali-masked-softmax-loss.full.en.md), [Alibaba re-ranking consistency principles](../multi-task-learning/2025-04-08-ali-rerank-consistency.full.en.md).

Intro of this issue:

(1) All 3 papers in this issue are worth reading. The techniques they use may not be very new, but the devil is in the details.

(3) The highlight of the second paper is how it handles "invalid tokens", making positive/negative sample learning in generative recommendation more effective.

**The devil is in the details: something worth reminding ourselves of again and again at work!**

Masked Softmax Loss: Mitigating the Impact of Fictitious Items in Generative Recommendation

Link: https://arxiv.org/pdf/2504.04517

Company: Alibaba

Area: generative recommendation

Overview:

This paper proposes a new loss, Masked Softmax Loss (MSL). By masking invalid tokens (the tokens corresponding to fictitious item descriptions) in the loss computation, it avoids penalizing those tokens and focuses better on distinguishing positive from negative items. Concretely, when softmax computes probabilities, the denominator leaves out the invalid tokens.

Fictitious items are items that do not actually exist in the recommender system, i.e. item descriptions that are not part of the system's item set. They are usually "imagined" or "made up" by the large language model (LLM) during generation, rather than real items the user actually interacted with.

Treating all fictitious item descriptions as negatives, even though some may be semantically similar to the positive item (e.g. "Iron Man: AI Rebellion" vs. "Iron Man 1"), feeds misleading training signals to the LLM.

Takeaways:

Penalizing things carelessly causes trouble. Invalid tokens are not the output we want, but we should not penalize them; we should route around them, e.g. avoid them with other post-processing methods.

Credibility: offline

Recommendation: worth reading.
