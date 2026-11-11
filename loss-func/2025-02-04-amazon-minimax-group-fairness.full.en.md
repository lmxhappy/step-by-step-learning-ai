# Recommendation papers (2025.1.31): Amazon uses DRO to improve minimax group fairness in sequential recommendation
> [中文版](2025-02-04-amazon-minimax-group-fairness.md)

> Source: WeChat post https://mp.weixin.qq.com/s/k7nEXZJqXf6nJ96EkpgPzw, published 2025-02-04.
> Split from the roundup "Recommendation algorithm papers (2025.1.31)"; same issue: [Alibaba DRP search behavior modeling](../behavior-sequence-modeling/2025-02-04-ali-drp-search-behavior.full.en.md), [Meta HyperZero fusion-formula auto-tuning](../multi-task-learning/2025-02-04-meta-hyperzero.full.en.md), [Amazon minimax group fairness](../loss-func/2025-02-04-amazon-minimax-group-fairness.full.en.md).

Improving Minimax Group Fairness in Sequential Recommendation

Link: https://arxiv.org/pdf/2501.18117

Company: Amazon

Area: behavior sequences + fairness

Overview: This paper tackles the problem that recommendations work poorly for specific user groups in sequential recommendation, focusing on minimax group fairness. The authors explore three methods related to DRO (an optimization approach designed to handle uncertainty in optimization problems). Experiments show that DRO methods are generally better than standard training, CVaR DRO works best, and it adapts better to scenarios with intersecting user groups.

CVaR: the average loss when the loss exceeds VaR at a given confidence level. It looks at the average of the losses beyond VaR, i.e., it answers "when the loss exceeds VaR, what is the average loss?" For example, in the example above, if CVaR is 800,000 yuan, it means that when the loss exceeds 500,000 yuan, the average loss will reach 800,000 yuan.

Credibility: offline

Recommendation: read if you have time
