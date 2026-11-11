# Recommendation papers (2024.12.24): Microsoft treats the recommender as a new language inside the LLM (RSLLM)
> [中文版](2024-12-24-microsoft-rsllm.md)

> Source: WeChat post https://mp.weixin.qq.com/s/8UzNMNhhiqc8HL3SYfQ3GQ, published 2024-12-24.
> Split from the roundup "Recommendation papers (2024.12.24)"; same issue (4 papers): [Huawei Scenario-Wise Rec multi-scenario benchmark](../cross-domain-recommendation/2024-12-24-huawei-scenario-wise-rec-benchmark.full.en.md), [Microsoft RSLLM](../llm4rec/2024-12-24-microsoft-rsllm.full.en.md), [Snap GraphHash embedding-table compression](../ml-infra/2024-12-24-snap-graphhash.full.en.md), [Huawei SyNeg LLM-synthesized hard negatives](../deep-retrieval/2024-12-24-huawei-syneg-llm-hard-negatives.full.en.md).

Towards a Unified Paradigm: Integrating Recommendation Systems as a New Language in Large Models

Link: https://arxiv.org/pdf/2412.16933

Company: Microsoft

Area: LLM-based recommendation

Overview: This paper explores using large language models (LLMs) for sequential recommendation, which aims to predict a user's future interactions from their past behavior. It introduces a new concept, "integrating recommendation systems as a new language in large models" (RSLLM), which combines the strengths of traditional recommenders and LLMs. RSLLM uses a distinctive prompting method that fuses the ID-based item embeddings from traditional recommendation models with textual item features. It treats the user's sequential behavior as a distinct language and uses a projection layer to align the ID embeddings with the LLM's input space.

Credibility: Offline

Recommendation: Worth reading
