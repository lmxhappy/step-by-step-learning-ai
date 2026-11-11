# Recommendation papers (2024.11.25): Huawei LIBER uses an LLM to summarize lifelong user behavior partition by partition
> [中文版](2024-11-25-huawei-liber-llm-lifelong.md)

> Source: WeChat post https://mp.weixin.qq.com/s/Tui8ZesqJjpCHRf2hUCRig, published 2024-11-25.
> Split from the roundup "Recommendation papers (2024.11.25)"; same issue (2 papers): [Huawei MIRRN multi-granularity interest retrieval and refinement](../behavior-sequence-modeling/2024-11-25-huawei-mirrn-multi-granularity-interest.full.en.md), [Huawei LIBER LLM-based lifelong behavior modeling](../behavior-sequence-modeling/2024-11-25-huawei-liber-llm-lifelong.full.en.md).

LIBER: Lifelong User Behavior Modeling Based on Large Language Models

Link: https://arxiv.org/pdf/2411.14713

Company: Huawei

Area: long behavior sequence modeling

Overview: The user behavior sequence is cached; once the cache length reaches a threshold it becomes a partition, and an LLM analyzes that partition to produce a summary of it, and also analyzes the relationship between this partition and earlier partitions. Looking at the cases given in the paper, it feels like an LLM can be an assistant not only for humans but also for the recommendation model, and a helpful one. In the figure below, the blue box is the question sent to the LLM and the red box is the LLM's answer. The answer does look like it adds value.

Credibility: production

Recommendation: worth a look

