# Recommendation Paper Pick (2025.7.19): Pinterest delivers quality again: GPT-style modeling of behavior sequences

> [中文版](2025-07-19-pinterest-pinfm.md)

### Paper 1: PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform
### Link: https://arxiv.org/pdf/2507.12704
### Company: Pinterest
### Area: behavior sequences + LLM

## Overview:

The paper trains a module for interest modeling from behavior sequences. The module is pretrained on its own, then plugged into the main recommendation model and fine-tuned.

(1) Pretraining

Data: two years of user activity data are collected (timestamp, action type, page type, item ID). Sequence length is capped at 16,000 and split into segments a few hundred long.

Model: a GPT-2 architecture (with Pre-LN), trained for next-token prediction with an InfoNCE loss. It adds multi-token prediction (Multi Token Loss) and future-token prediction (Future-Token Loss) to capture both short-term and long-term changes in user interest.

Optimizations: ID embeddings are used instead of high-dimensional pretrained embeddings, lowering storage and compute cost; for the huge vocabulary (hundreds of millions of item IDs), a contrastive learning loss is used.

(2) Fine-tuning

The pretrained model is integrated into the downstream ranking model (e.g. DLRM or DCN). Early fusion (adding the candidate item to the user sequence) strengthens the capture of user-item interactions. Fine-tuning adds the pretraining loss to the ranking model's loss, and the learning rate is set to 1/10 of the ranking model's, to preserve pretrained knowledge. Cold-start optimization: candidate item ID randomization and item-age-based dropout improve recommendation performance on new items.

(3) Inference

DCAT, through deduplication, KV cache reuse, fixed sequence length, skipping redundant computation and Triton kernel optimization, significantly reduces PinFM's inference latency, turning it from potentially "very long" into efficient and controllable.

The model caches the K and V of every layer in the self-attention over the user behavior sequence; afterwards it only needs to do cross attention with the candidate item's per-layer KV, which reduces inference time.

In addition, the embedding tables are hosted on a CPU cluster and transformer inference runs on a GPU cluster, with inference routing passing data efficiently to avoid GPU memory bottlenecks.

Also, the embedding tables are quantized from float16 to int4, shrinking their size to 31.25%, reducing data transfer and lowering inference latency by 7% (due to less embedding I/O load).

After PinFM is integrated into the main recommendation models (e.g. the ranking models of Home Feed and Related Items Feed), processing one query has to fit within a latency of a few hundred milliseconds.

A/B results:

(1) Home Feed: overall Saves +1.20%, page Saves +2.60%, fresh-item Saves +8.20%;

(2) Related Items Feed: overall Saves +0.90%, page Saves +2.10%, fresh-item Saves +7.50%.

## Takeaways:

(1) Modeling user behavior sequences in a recommender system as a language-model-style sequence prediction task is a really nice idea. In particular, multi-token prediction and future-token prediction can capture both short-term and long-term changes in user interest.

(2) Integrating the pretrained model into the downstream ranking model can significantly improve recommendation quality. In particular, early fusion and the cold-start optimizations improve recommendation performance on new items.

## Credibility: production

## Recommendation: practically useful

<!-- Figure for this article: the page's image element returned only a placeholder, so the image link could not be retrieved -->
