# Recommendation papers (2025.1.31): Meta HyperZero: auto-tuning fusion-formula hyperparameters with hourly feedback
> [中文版](2025-02-04-meta-hyperzero.md)

> Source: WeChat post https://mp.weixin.qq.com/s/k7nEXZJqXf6nJ96EkpgPzw, published 2025-02-04.
> Split from the roundup "Recommendation algorithm papers (2025.1.31)"; same issue: [Alibaba DRP search behavior modeling](../behavior-sequence-modeling/2025-02-04-ali-drp-search-behavior.full.en.md), [Meta HyperZero fusion-formula auto-tuning](../multi-task-learning/2025-02-04-meta-hyperzero.full.en.md), [Amazon minimax group fairness](../loss-func/2025-02-04-amazon-minimax-group-fairness.full.en.md).

HyperZero: A Customized End-to-End Auto-Tuning System for Recommendation with Hourly Feedback

Link: https://arxiv.org/pdf/2501.18126

Company: Meta

Area: rerank fusion formula

Overview: Automated tuning of the hyperparameters of the fusion formula (the paper calls it the value model stage), using hourly system feedback. It first handles the non-i.i.d. noisy data with special hourly signals, then explores the hyperparameter space with a zeroth-order optimization method combining Gaussian processes and Thompson sampling, and uses asynchronous parallel exploration to find the optimum faster, achieving efficient tuning.

Credibility: production

Recommendation: worth reading
