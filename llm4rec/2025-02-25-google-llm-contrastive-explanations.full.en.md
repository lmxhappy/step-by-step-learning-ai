# Recommendation papers (2025.2.25): Google LR-Recsys uses LLM-generated positive and negative contrastive explanations to boost a black-box recommender
> [中文版](2025-02-25-google-llm-contrastive-explanations.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484189&idx=1&sn=c256db252d5aa7ce3629b8c45953a2d5&scene=21#wechat_redirect, published 2025-02-25.
> Split from the roundup "Recommendation papers, fast! The latest practical results from four big companies"; same issue: [Meta unified semantic and ID representation](../sid/2025-02-25-meta-unified-semantic-id.full.en.md), [Google LR-Recsys contrastive explanations](../llm4rec/2025-02-25-google-llm-contrastive-explanations.full.en.md), [Huawei SCCDR cross-domain contrastive learning](../cross-domain-recommendation/2025-02-25-huawei-separated-contrastive-cdr.full.en.md), [Alibaba ESANS semantic-aware negative sampling](../deep-retrieval/2025-02-25-ali-esans-negative-sampling.full.en.md).

Intro of this issue:

This post focuses on frontier research in recommender systems, covering LLM recommendation, cross-domain recommendation, and embedding-based retrieval. It walks through four papers from well-known companies and gives takeaways and recommendation levels.

The Blessing of Reasoning: LLM-Based Contrastive Explanations in Black-Box Recommender Systems

Link: https://arxiv.org/pdf/2502.16759

Company: Google

Area: LLM recommendation

Overview: Combining large language models (LLMs) with deep neural networks (DNNs) improves performance on both fronts at once.

LR-Recsys introduces a contrastive explanation generator that uses LLMs to produce human-readable positive explanations (why a consumer might like a product) and negative explanations (why a consumer might not like it). These explanations are embedded with a fine-tuned autoencoder and fed into the DNN for fusion.

Credibility: offline

Recommendation: worth reading
