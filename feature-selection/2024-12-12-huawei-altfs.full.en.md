# Recommendation papers (2024.12.12): Huawei AltFS: feature selection for recommendation models with an LLM
> [中文版](2024-12-12-huawei-altfs.md)

> Source: WeChat post https://mp.weixin.qq.com/s/B33h3N0mVXTEPYl3Snr3oQ, published 2024-12-12.
> Split from the roundup "Recommendation papers (2024.12.12)"; same issue: [Meta: LLM-distilled user interests](../llm4rec/2024-12-12-meta-preference-discerning.full.en.md), [Huawei AltFS feature selection](../feature-selection/2024-12-12-huawei-altfs.full.en.md).

AltFS: Agency-light Feature Selection with Large Language Models in Deep Recommender Systems

Link: https://arxiv.org/pdf/2412.08516

Company: Huawei

Area: feature selection + LLM

Overview: Use an LLM to pick the important features from the feature set. The paper uses 3 well-known public datasets and has GPT-4 and similar models report feature importance, with good results. GPT-4 and other LLMs are already very familiar with these datasets. But for private production data that the LLM is not familiar with, it probably won't work as well; then you would have to find a way to feed in the private data, train the LLM, and only then have the large model select features for us.

Credibility: offline

Recommendation: read if you have time
