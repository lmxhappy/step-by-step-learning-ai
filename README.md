<p align="center"><img src="./imgs/banner.png" alt="Step-by-Step Learning AI — Industrial Recommender Systems Paper Deep Dives" width="100%"></p>


[![GitHub stars](https://img.shields.io/github/stars/lmxhappy/step-by-step-learning-ai?style=social)](https://github.com/lmxhappy/step-by-step-learning-ai)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=lmxhappy.step-by-step-learning-ai)](https://github.com/lmxhappy/step-by-step-learning-ai)
[![Hits](https://hits.sh/github.com/lmxhappy/step-by-step-learning-ai.svg)](https://github.com/lmxhappy/step-by-step-learning-ai/)

[![English](https://img.shields.io/badge/README-English-0A66C2?style=for-the-badge)](README.md) [![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-c0392b?style=for-the-badge)](README.zh-CN.md)

Paper write-ups on industrial **recommendation, ads and search** systems: what the method does, what it gained in production A/B tests, and where the claims are weaker than they look.

Most notes are written in Chinese for the WeChat account **稳扎稳打学AI**; a growing set has full English versions, listed below. Each English write-up sits next to its Chinese original with the suffix `.en.md`.

## English write-ups

| Paper | Company | What it's about |
|---|---|---|
| [TransX](./behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/linkedin-transx.en.md) | LinkedIn | Compress the user's behavior sequence offline into 64 vectors stored as KV; online, candidates only attend to them. CTR +6.02% over an HSTU-style model. |
| [PAIR](./deep-retrieval/linkedin-pair.en.md) | LinkedIn | Embedding retrieval for Premium people recommendation: a platform-wide member embedding replaced by one fine-tuned on the surface's own clicks. CTR +7.5%, Premium retention +2.1%. |
| [Pin-SCALE](./sid/pinterest-pin-scale.en.md) | Pinterest | Wiring Semantic IDs into two-tower retrieval for cold start. Fresh repins +3.67%. |

## Topics

All notes are organized by topic folder. Notes not listed above are in Chinese only; the full per-paper index is in the [Chinese README](README.zh-CN.md).

| Layer | Topics |
|---|---|
| Recommendation paradigm | [Generative recommendation](./generative-recommendation/) · [LLM4Rec](./llm4rec/) |
| Representation | [Semantic IDs](./sid/) · [Representation quality](./representation-quality/) |
| Model architecture | [Behavior sequence modeling](./behavior-sequence-modeling/) · [Feature interaction](./feature-cross/) · [Deep retrieval](./deep-retrieval/) |
| Data & features | [Feature selection](./feature-selection/) |
| Training objective | [Loss functions](./loss-func/) |
| Scenarios | [Cross-domain](./cross-domain-recommendation/) · [Item-to-item](./item-to-item/) |
| Problems | [Long tail](./long-tail/) |
| Engineering | [ML infra](./ml-infra/) · [Auto research](./auto-research/) |

**Author:** Mingxing Liu · WeChat `lmxhappy` · [Zhihu](https://zhuanlan.zhihu.com/stupid-ai). If this repo helps you, a ⭐ helps others find it.

This is a notes repo, not a code or reproduction repo.
