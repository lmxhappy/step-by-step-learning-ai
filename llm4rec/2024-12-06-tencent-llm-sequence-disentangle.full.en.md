# Recommendation papers (2024.12.6): Tencent fuses LLM text and behavior sequences via MoE to ease item cold start
> [中文版](2024-12-06-tencent-llm-sequence-disentangle.md)

> Source: WeChat post https://mp.weixin.qq.com/s/v7NZ4K9t520KMFIZc8n3PA, published 2024-12-06.
> Split from the roundup "Recommendation papers (2024.12.6)"; same issue (2 papers): [Salesforce: sequence-graph alignment](../behavior-sequence-modeling/2024-12-06-salesforce-graph-sequence-alignment.full.en.md), [Tencent: LLM-empowered sequential rec](../llm4rec/2024-12-06-tencent-llm-sequence-disentangle.full.en.md).

Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models

Link: https://arxiv.org/pdf/2412.04107

Company: Tencent

Area: behavior sequence + LLM + cold start

Overview: Trains a behavior sequence model and an LM text encoder separately, then combines the two to build 3 experts trained together in an MoE fashion, hoping the text information can ease the item cold-start problem.

Credibility: offline

Recommendation: read if you have time
