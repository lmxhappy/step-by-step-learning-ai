# Recommendation papers (2025.2.5): Xiangnan He's team uses influence functions to handle delayed feedback in CVR prediction
> [中文版](2025-02-05-academic-delayed-feedback-influence.md)

> Source: WeChat post https://mp.weixin.qq.com/s/iXXwsZ46YhkqLlZ-OMbvNg, published 2025-02-05.
> Split from the roundup "Recommendation papers (2025.2.5)"; same issue: [Kuaishou COPF multi-behavior recommendation](../multi-task-learning/2025-02-05-kuaishou-copf-multi-behavior.full.en.md), [Xiangnan He's team: deliberative preference alignment for LLM recommendation](../llm4rec/2025-02-05-academic-llm-deliberative-preference.full.en.md), [Xiangnan He's team: influence functions for delayed feedback in CVR](../loss-func/2025-02-05-academic-delayed-feedback-influence.full.en.md).

Addressing Delayed Feedback in Conversion Rate Prediction via Influence Functions

Link: https://arxiv.org/pdf/2502.01669

Company: Xiangnan He.

Area: ad CVR prediction

Overview: This paper focuses on the delayed feedback problem in conversion rate (CVR) prediction and proposes an influence-function-based delayed feedback modeling framework (IF-DFM). The core idea is to use influence functions to estimate how new data and delayed conversion data affect the model parameters, and to modify the parameters directly, avoiding retraining. In addition, it turns the inverse Hessian-vector product computation into a finite-sum quadratic programming problem and uses stochastic optimization to make the computation more efficient. Two papers published on the same day: my guess is that Prof. He approved both for release at once.

Credibility: offline

Recommendation: read if you have time
