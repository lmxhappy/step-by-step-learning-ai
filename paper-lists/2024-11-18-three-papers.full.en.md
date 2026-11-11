# Recommendation paper list (2024.11.18)

> [中文版](2024-11-18-three-papers.md)

> Source: WeChat post https://mp.weixin.qq.com/s/-UD2-yNlJtuYUJYGRQkagA, published 2024-11-18.

1 KuaiFormer: Transformer-Based Retrieval at Kuaishou

Link: https://arxiv.org/pdf/2411.10057

Company: Kuaishou. From the team of Guorui Zhou, author of DIN.

Direction: retrieval + transformer

Summary: uses transformers for retrieval. Because the item pool size naturally demands high efficiency from retrieval models, transformers had not been deployed in real-time retrieval. This work deploys it, at billion-item scale, with multiple interests and long sequences.

Results: production.

Rating: worth a look.

2 InterFormer: Towards Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction

Link: https://arxiv.org/pdf/2411.09852

Company: Meta

Direction: ranking + multimodal

Summary: improves how heterogeneous data works in multimodal models. Multimodality has two challenges: first, not enough interaction among the different kinds of heterogeneous data; second, each kind of heterogeneous data gets aggregated too early. This work tries to solve both. Interested readers can see how exactly.

Results: production

Rating: worth a look

3 Residual Multi-Task Learner for Applied Ranking

Link: https://arxiv.org/pdf/2411.09705

Company: Shopee

Direction: search ranking + multi-task

Summary: across tasks, the outputs of intermediate layers are added to other tasks.

Results: production

Rating: worth a look
