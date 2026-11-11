# Recommendation papers (2025.4.16): Amazon CSPLADE, learned sparse retrieval with causal LLMs
> [中文版](2025-04-16-amazon-csplade.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484494&idx=1&sn=4765dee4aa47534d3892f6f9b9fdb89c&scene=21#wechat_redirect, published 2025-04-16.
> Split from the roundup "Recommendation papers (2025.4.16): Pinterest & Amazon"; same issue: [Pinterest PinRec generative retrieval](../generative-recommendation/2025-04-16-pinterest-pinrec.full.en.md), [Amazon CSPLADE sparse retrieval](../deep-retrieval/2025-04-16-amazon-csplade.full.en.md).

CSPLADE: Learned Sparse Retrieval with Causal Language Models

Link: https://arxiv.org/pdf/2504.10816

Company: Amazon

Area: learned sparse retrieval (LSR)

Overview:

The paper addresses two main challenges of using LLMs for LSR: training instability, and suboptimal performance caused by unidirectional attention.

What is learned sparse retrieval (LSR)?

A pretrained language model (e.g. BERT or an LLM) learns to produce a vocabulary-sized sparse vector, where each dimension gives the "importance" or "impact" of one vocabulary term. Terms with non-zero importance (above a threshold) are used to build an inverted index for efficient retrieval, and the index is much smaller than for dense retrieval.

Method: a pretrained causal LLM (e.g. LLAMA-3.1-8B) is used as the backbone to produce vocabulary-sized sparse vectors that fit an inverted index for efficient retrieval. Training pipeline:

(1) Adaptation stage: lightweight pretraining on unlabeled text, optimizing a combination of the causal language modeling loss and a ReLU adaptation loss.

(2) Contrastive training stage: contrastive learning with the InfoNCE loss, combined with hard negative mining and FLOPs regularization to increase sparsity.

Takeaways: it uses the strong understanding ability of large models to give match-stage retrieval a new lease of life.

Credibility: offline

Recommendation: read if you have time
