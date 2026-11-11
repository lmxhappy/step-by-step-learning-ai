# Recommendation paper list (2024.11.04)

> [中文版](2024-11-04-two-papers.md)

> Source: WeChat post https://mp.weixin.qq.com/s/cunko32RxO5bvXuxTkawzQ, published 2024-11-03.

1. Beyond Utility: Evaluating LLM as Recommender

a. Link: https://arxiv.org/pdf/2411.00331

b. Company: Huawei

c. Direction: LLM-based recommendation. On evaluating LLM recommenders, it proposes 4 evaluation dimensions and a multi-dimensional evaluation framework. It finds that LLM recommenders do well on ranking tasks with prior knowledge and shorter input histories, perform better on re-ranking tasks, and beat traditional models on several dimensions, but have a significant candidate position bias problem, and some models more often generate items that don't exist. Side note: this paper gives you a full picture of where LLMs are, their strengths and weaknesses, and the problems they face.

2. DivNet: Diversity-Aware Self-Correcting Sequential Recommendation Networks

a. Link: https://arxiv.org/pdf/2411.00395

b. Company: Alibaba

c. Direction: rerank + diversity. Uses self-attention networks for a rerank that balances diversity of the final item list, overall user utility, and exposure bias. The title doesn't call it rerank but collective recommendation; I honestly wasn't familiar with that term. Forgive my ignorance, still need to work harder.
