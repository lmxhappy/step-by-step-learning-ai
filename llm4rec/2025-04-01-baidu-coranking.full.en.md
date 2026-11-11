# Recommendation papers (2025.4.1): Baidu's CoRanking, collaborative ranking with small and large models
> [中文版](2025-04-01-baidu-coranking.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484391&idx=1&sn=ef838380260b349648ae9d9bc69b52ab&scene=21#wechat_redirect, published 2025-04-01.
> Split from the roundup "Recommendation papers (2025.4.1): industrial style all the way! Work from Meta, Baidu and Microsoft"; same issue: [Meta position-biased item selection](../deep-retrieval/2025-04-01-meta-position-biased-retrieval.full.en.md), [Microsoft Wasserstein reranking](../loss-func/2025-04-01-microsoft-wasserstein-rerank.full.en.md), [Baidu CoRanking small/large model collaborative ranking](../llm4rec/2025-04-01-baidu-coranking.full.en.md).

CoRanking: Collaborative Ranking with Small and Large Ranking Agents

Link: https://arxiv.org/pdf/2503.23427

Company: Baidu

Area: rerank

Overview:

The core idea is a collaborative ranking framework called CoRanking, as follows:

(1) Small model for the global pass. First use a small reranker to pre-rank all candidate passages, moving relevant passages toward the top of the list.

(2) Large model for the local pass. Then use an LLM listwise reranker to rerank the top passages output by the small reranker, improving overall ranking efficiency.

(3) Bias correction. To address the position bias of LLM listwise rerankers, it introduces a passage order adjuster trained with reinforcement learning, which reorders the passages output by the small reranker to match the LLM's preferred passage order, achieving search result ranking that is both efficient and effective.

Takeaways: Huawei's paper says LLMs empowering recommendation algorithms is useful, but improving efficiency is the key to deployment. This paper is one example of deployment. Industrial style all the way.

Credibility: offline

Recommendation: read if you have time. If you don't have time, the overview and takeaways are enough.
