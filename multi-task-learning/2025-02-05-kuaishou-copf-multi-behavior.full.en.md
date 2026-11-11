# Recommendation papers (2025.2.5): Kuaishou tackles multi-behavior recommendation from a combinatorial optimization perspective
> [中文版](2025-02-05-kuaishou-copf-multi-behavior.md)

> Source: WeChat post https://mp.weixin.qq.com/s/iXXwsZ46YhkqLlZ-OMbvNg, published 2025-02-05.
> Split from the roundup "Recommendation papers (2025.2.5)"; same issue: [Kuaishou COPF multi-behavior recommendation](../multi-task-learning/2025-02-05-kuaishou-copf-multi-behavior.full.en.md), [Xiangnan He's team: deliberative preference alignment for LLM recommendation](../llm4rec/2025-02-05-academic-llm-deliberative-preference.full.en.md), [Xiangnan He's team: influence functions for delayed feedback in CVR](../loss-func/2025-02-05-academic-delayed-feedback-influence.full.en.md).

Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation

Link: https://arxiv.org/pdf/2502.02232

Company: Kuaishou

Area: multi-behavior recommendation

Overview: This paper focuses on multi-behavior recommendation and points out that mainstream methods fall short in behavior fusion and task coordination. It proposes COPF, a framework built from a combinatorial optimization perspective, with two parts: COGCN and DFME. COGCN imposes constraints on user behavior patterns at different stages to solve the combinatorial optimization problem of multi-behavior fusion; DFME improves forward and backward propagation at the feature and label levels to coordinate the correlations among the multi-behavior prediction tasks. Specifically, in the multi-behavior prediction stage, DFME improves the forward and backward passes at both the feature level and the label level: contrastive learning and behavior-fitting expert generation reduce feature distribution bias, and an improved gating mechanism at aggregation time avoids negative transfer caused by differences in label distributions.

Credibility: offline

Recommendation: worth reading
