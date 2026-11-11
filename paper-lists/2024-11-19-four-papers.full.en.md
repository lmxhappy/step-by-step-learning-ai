# Recommendation paper list (2024.11.19)

> [中文版](2024-11-19-four-papers.md)

> Source: WeChat post https://mp.weixin.qq.com/s/xNpSP1hhr5jJw3R_XegXdg, published 2024-11-19.

1 QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou

Link: https://arxiv.org/pdf/2411.11739

Company: Kuaishou. Guorui Zhou's team.

Direction: ranking + multimodal

Overview: transforms multimodal embeddings before using them in the ranking model. Existing work has two problems: first, CV/NLP models have different training objectives from the ranking model; second, multimodal embeddings are not learnable. This work aims to ease both.

Results: production

Rating: worth a look

2 Collaborative Contrastive Network for Click-Through Rate Prediction

Link: https://arxiv.org/pdf/2411.11508

Alibaba Taobao.

Direction: ranking + post-click recommendation (TIR, trigger-induced recommendation)

Overview: uses contrastive learning to build the relationship between the trigger item and the target item. This is the first time contrastive learning has been brought into TIR. In Taobao's online A/B test, CTR rose 12.3% and orders rose 12.7%, which shows the power of contrastive learning.

Results: production

Rating: worth a look

3 All-domain Moveline Evolution Network for Click-Through Rate Prediction

Link: https://arxiv.org/pdf/2411.11502

Company: Alibaba Taobao. Same authors as the previous paper; two papers released in one day, that's just unfair.

Direction: ranking + intent recognition

Overview: previous intent recognition was item-level and single-scenario, i.e. in the CTR model, the user's intent toward the target item is modeled using data from a single scenario only. This paper wants to collect the user's data across all scenarios to model the user's real purpose. The idea is quite naive, because it's very hard to get a user's intent from a single scenario. Only by stringing together all of a user's behavior across the app can you understand their intent better and recommend better for them.

Results: production

Rating: worth a look

4 Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning

Link: https://arxiv.org/pdf/2411.11225

Company: Kuaishou

Direction: ranking + cold start

Overview: my ability is limited; spent ten minutes and didn't understand it, god.

Results: production

Rating: read if you have time. Those working on cold start should take a look.
