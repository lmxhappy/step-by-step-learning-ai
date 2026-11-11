# Recommendation paper list (2024.11.05)

> [中文版](2024-11-05-three-papers.md)

> Source: WeChat post https://mp.weixin.qq.com/s/WundhKOYdM93hyuwm6qtdg, published 2024-11-04.

1. LLM4PR: Improving Post-Ranking in Search Engine with Large Language Models

a. Link: https://arxiv.org/pdf/2411.01178

b. Company: Kuaishou

c. Direction: search + post-ranking + LLM. Applies LLMs to the post-ranking stage of search.

d. Credibility: offline validation. MovieLens-1M & KuaiSAR

2. Enhancing ID-based Recommendation with Large Language Models

a. Link: https://arxiv.org/pdf/2411.02041

b. Company: Meituan

c. Direction: LLM. Strengthens an LLM's ability to understand ID data and ID-based recommendation tasks. Put simply, a normal LLM takes text in and puts text out; this paper wants the LLM to take in and output IDs, including user IDs and item IDs. A fine-tuned LLM then generates augmented item IDs, and this augmented data further improves the model.

d. Credibility: offline validation

3. LinRec: Linear Attention Mechanism for Long-term Sequential Recommender Systems

a. Link: https://arxiv.org/pdf/2411.01537

b. Company: Ant Group

c. Direction: long behavior sequences + attention. LinRec has linear complexity while keeping the properties of attention. Left for interested readers to dig into.

d. Credibility: offline validation.

e. Code: https://github.com/Applied-Machine-Learning-Lab/LinRec

Afterword: so many LLM papers.
