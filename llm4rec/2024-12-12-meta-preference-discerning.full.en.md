# Recommendation papers (2024.12.12): Meta uses an LLM to distill user interests from behavior sequences to enhance generative retrieval
> [中文版](2024-12-12-meta-preference-discerning.md)

> Source: WeChat post https://mp.weixin.qq.com/s/B33h3N0mVXTEPYl3Snr3oQ, published 2024-12-12.
> Split from the roundup "Recommendation papers (2024.12.12)"; same issue: [Meta: LLM-distilled user interests](../llm4rec/2024-12-12-meta-preference-discerning.full.en.md), [Huawei AltFS feature selection](../feature-selection/2024-12-12-huawei-altfs.full.en.md).

Preference Discerning with LLM-Enhanced Generative Retrieval

Link: https://arxiv.org/pdf/2412.08604

Company: Meta

Area: behavior sequence + LLM

Overview: An LLM distills multiple user interests from the user's behavior sequence, which are then applied to behavior sequence modeling. The key is distilling user interests from the sequence: concatenate the information of the sequence elements, such as their descriptions, with the user's review text and so on, feed it into the LLM, and let it summarize the user's interests. The distilled interests are also text; the text is turned into embeddings with a multimodal encoder, which then improve the downstream behavior sequence modeling. Using LLMs to distill user interests and intent is a very hot approach; many recent papers are doing it, and it appears to work. This is easy to understand: recommendation models are fairly shallow and struggle to model user intent, while when using ChatGPT we clearly sense that its level of intelligence is high and its understanding of user intent is better, and in the foreseeable future it will get better as large language models develop.

Credibility: offline

Recommendation: worth reading
