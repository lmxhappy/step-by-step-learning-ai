# Recommendation papers (2025.4.15): Meta URM, one model unifying retrieval, ranking and other recommendation tasks
> [中文版](2025-04-15-meta-urm-unified-model.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484486&idx=1&sn=6687984fa2dd4d718ea0c55554f2747d&scene=21#wechat_redirect, published 2025-04-15.
> Split from the roundup "Recommendation papers (2025.4.15): 3 papers from big companies that are not about large models"; same issue: [Meta URM unified recommendation model](../deep-retrieval/2025-04-15-meta-urm-unified-model.full.en.md), [Meta FROG multimodal friend recommendation](../graph-recommendation/2025-04-15-meta-frog-friend-rec.full.en.md), [Meta unsupervised data enhancement for sequential recommendation](../behavior-sequence-modeling/2025-04-15-meta-unsupervised-seq-enhancement.full.en.md).

Unified Recommendation Model (URM): Retrieval, Ranking and Beyond

Link: https://arxiv.org/pdf/2504.09631

Company: Meta

Area: retrieval

Overview:

The paper proposes a unified recommendation model that can do retrieval, ranking and other recommendation tasks at the same time. The core is modeling everything with a unified input representation and a Transformer architecture.

(1) Input representation: for each user, user-side information is described by one [U] token (which can include user features, context, etc.), plus multiple candidate items, each corresponding to a [C] token. The model processes these token sequences to produce the user's click probability.

(2) Encoder/generator: both the encoder and the generator are Transformers. The encoder processes the full token sequence and produces context-aware embeddings; the generator uses these embeddings to produce click probabilities for the target item tokens. The generator is autoregressive: based on the tokens generated so far, it predicts the click probability of the next item.

(3) Multi-objective recommendation: it supports diverse recommendation objectives (e.g. diversity, personalization) at the same time, producing recommendation probabilities for the corresponding items through different objective tokens. The objective token is [U], [C] or another specific objective. Each objective corresponds to a different token (e.g. "recommend similar items" or "recommend diverse items"), which generates the corresponding items.

Takeaways: a unified model that solves cross-task transfer is a very good direction.

Credibility: offline

Recommendation: worth reading
