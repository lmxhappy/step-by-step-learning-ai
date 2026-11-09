# Recommendation papers (2026.1.23): ByteDance feature interaction method, GMV per user +6%

> [中文版](2026-01-23-bytedance-onetrans.md)

1 OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender

Link: https://arxiv.org/pdf/2510.26104

Company: ByteDance

Idea: the more crossing the better

Area: feature interaction

Overview:

This paper proposes a model that unifies sequence modeling and feature interaction, called OneTrans. It models both feature interaction and sequence modeling with "One Transformer", and uses a pyramid attention mechanism to solve the performance problem of Transformers on long sequences, which improves results.

Specifically:

(1) Sequence modeling

The user behavior sequence contains a sequence of clicked items and the corresponding target items. The clicked items and the target item are used as query and key/value respectively; query and key/value go through cross-attention, and an MLP produces a vector. At the same time, the interaction scores between query and key/value are taken as weights over the clicked-item sequence, and a weighted sum over the clicked items gives the representation of the user behavior sequence.

This way, the user behavior sequence is crossed with the target item, and feature crossing also happens inside sequence modeling (that is, feature interaction is folded into sequence modeling).

(2) Feature interaction modeling

All non-sequence features of a sample, including static features (gender, age, occupation, region) and context features (time, device, location), are first turned into a set of tokens by a Tokenizer module, then fed together into one Transformer. A stacked attention structure models sequence information and feature interaction at the same time, and a cross-attention layer produces the final item representation. The benefit is that feature interaction and sequence modeling happen inside the same Transformer, avoiding the suboptimal result of separating them into two stages (for example, feature interaction first, then sequence modeling). In other words, modeling feature interaction and sequences together in one model captures their interplay more effectively.

In particular, the Transformer is expensive, so to cut compute it uses pyramid stacking: each layer of the multi-layer Transformer uses a pooling function at a different scale, and every layer's output serves as a representation of the sequence features for the later cross-attention. Through gradual compression, information is condensed step by step from the long sequence into the non-sequence tokens.

After this description, readers may ask: what is good about the pyramid architecture and this way of aggregating the sequence?

The answer:

(1) Efficiency for long sequences: attention on long sequences is very expensive, and pyramid-style multi-scale aggregation lowers the attention cost.

(2) Multi-granularity modeling: the pyramid stack naturally models information at different granularities, from fine to coarse, which helps the model understand user behavior patterns at different levels.

(3) Unified feature modeling: after the Tokenizer, static and context features are uniformly processed as tokens and enter the same Transformer, so feature interaction and sequence modeling are done together.

Note that tokenization here (turning non-sequence features into tokens) comes in two kinds:

1) Non-sequence tokenization: the options are group-wise or automatic splitting. The former splits the non-sequence features into groups, each with its own MLP, producing the corresponding tokens; the latter uses a single MLP to output one vector, which is then split evenly into several parts used as several tokens.

2) Sequence tokenization: each item's features go through an MLP and become a sequence token.

Other small tricks:

(1) Sequence behaviors: each behavior is represented by the sum of its ID embedding and category embedding;

(2) RMSNorm replaces LayerNorm to improve inference efficiency;

(3) Sequence position encoding: via the time difference of the behaviors to the target, i.e. the position encoding is determined by the time gap between the behavior and the target.

A/B: GMV per user +5.68%

Takeaways:

(1) Modeling feature interaction and sequences together in the same model gives better results. This paper was deployed on BytePlus and achieved +5.68% GMV per user.

(2) Pyramid stacking and multi-scale aggregation are effective ways to make long-sequence modeling efficient.

(3) Turning non-sequence features uniformly into tokens with a Tokenizer module is worth borrowing.

Credibility: production

Recommendation: practically useful
