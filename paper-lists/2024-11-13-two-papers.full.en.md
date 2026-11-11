# Recommendation paper list (2024.11.13)

> [中文版](2024-11-13-two-papers.md)

> Source: WeChat post https://mp.weixin.qq.com/s/KN41e96_U3XlyU0PtFNJRQ, published 2024-11-12.

1.Towards Automated Model Design on Recommender Systems

Link: https://arxiv.org/pdf/2411.07569

Company: Meta

Direction: recommender system optimization + AutoML.

Summary: proposes a paradigm that improves model efficiency: 2x better FLOPs efficiency, 1.8x better energy efficiency, and 1.5x better performance.

Credibility: production

Rating: read if you have time. Only a huge company like Meta feels such urgent pressure and motivation to solve performance problems.

2.AdaS&S: a One-Shot Supernet Approach for Automatic Embedding Size Search in Deep Recommender System

Link: https://arxiv.org/pdf/2411.07504

Company: Tencent

Direction: AutoML.

Summary: searches for dynamic feature embedding sizes. We decouple main-model training from the search process, which ensures reasonable parameters and stable embedding size allocation. An adaptive sampling method is designed for training the supernet, greatly improving its consistency. In the search stage a reinforcement learning (RL) based method is used to find the optimal AES result.

Credibility: offline validation

Rating: read if you have time. Likewise, an ordinary company can't use it and can't afford to play with it.
