# Recommendation papers (2025.4.1): Microsoft reranks with multi-level relevance data and a Wasserstein loss
> [中文版](2025-04-01-microsoft-wasserstein-rerank.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484391&idx=1&sn=ef838380260b349648ae9d9bc69b52ab&scene=21#wechat_redirect, published 2025-04-01.
> Split from the roundup "Recommendation papers (2025.4.1): industrial style all the way! Work from Meta, Baidu and Microsoft"; same issue: [Meta position-biased item selection](../deep-retrieval/2025-04-01-meta-position-biased-retrieval.full.en.md), [Microsoft Wasserstein reranking](../loss-func/2025-04-01-microsoft-wasserstein-rerank.full.en.md), [Baidu CoRanking small/large model collaborative ranking](../llm4rec/2025-04-01-baidu-coranking.full.en.md).

Wasserstein Reranking for Multi-level Context Retrieval

Link: https://arxiv.org/pdf/2503.23633

Company: Microsoft

Area: search

Overview:

Targeting the limitations of traditional ranking models trained on binary labels, it proposes a new method to improve search relevance. Traditional binary-label ranking has these limitations:

(1) It ignores subtle differences in relevance: apart from documents explicitly labeled relevant, all other documents are treated equally as negatives, which does not reflect the actual degree of relevance.

(2) It is sensitive to noise: binary labels cannot fully use the ordering information in the data, and are easily hurt by labeling errors.

Method:

(1) Multi-level ranking context generation:

Using queries from the MS MARCO dataset, prompt LLMs to generate four synthetic documents at different relevance levels. Generating them sequentially ensures relevance decreases step by step, and random sampling (e.g. of sentence length and difficulty) adds diversity. Labels assigned: {3, 2, 1, 0}, from perfectly relevant to irrelevant.

(2) Wasserstein distance loss:

Unlike InfoNCE's pairwise comparisons, the Wasserstein distance considers the relevance distribution over the whole document set. By comparing the true label distribution with the predicted score distribution, the model is optimized to rank documents more accurately.

Takeaways: making data more fine-grained is a good method. Models can be made fine-grained (decoupled), and so can data. Both should go finer.

Credibility: offline

Recommendation: worth reading
