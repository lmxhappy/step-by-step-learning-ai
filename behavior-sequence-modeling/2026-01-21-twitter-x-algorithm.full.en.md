# Recommendation algorithms (2026.1.21): Twitter's recommendation model is remarkably simple, understand it in 3 minutes

> [中文版](2026-01-21-twitter-x-algorithm.md)

1 Open-source model: https://github.com/xai-org/x-algorithm

Company: XAI

Idea: end-to-end

Area: Transformer ranking

Overview:

This codebase is the recommendation pipeline of Twitter's recommendation feed, covering retrieval, ranking and re-ranking. There is no pre-ranking stage.

(1) Retrieval

Multi-channel retrieval, for example:

1) retrieval based on content similarity (embedding retrieval);

2) retrieval based on a knowledge graph;

3) retrieval based on collaborative filtering;

4) retrieval based on popularity.

(2) Ranking model

Grok-1 recommendation version: built on Grok-1 with a modified attention scope. It is a Transformer model that does sequence modeling over all behavior on the platform. Specifically:

1) User behavior sequences, social relationships, post content, real-time context and other features all go in as tokens to the same Transformer and are modeled jointly;

2) The model does not use the traditional "user tower + item tower" structure; it directly models the joint sequence of the user and the candidate items;

3) **Multi-user, multi-candidate** inference: all candidate items retrieved for one user go in as one sequence, and when doing attention the candidates do not attend to each other, so they don't affect each other. In other words, it uses the Transformer's sequence modeling ability but achieves the effect of independent target attention.

1) and 3) are the highlights, especially 3). 2) is standard practice.

(3) Re-ranking

1) diversity re-ranking;

2) freshness re-ranking;

3) platform content policy compliance checks.

Takeaways:

(1) Multi-user, multi-candidate inference in one batch makes an inefficient Transformer efficient, so it can be deployed in production.

(2) Extremely end-to-end. No features are hand-crafted, and retrieval is only EBR (embedding-based retrieval).

(3) Very high requirements on platform content quality

Retrieval and re-ranking are relatively very simple. This requires the items on the platform to be of very high quality. The item corpus is at the million scale; my guess is that a lot of filtering and screening is done on items, keeping many high-quality ones.

Credibility: production

Recommendation: practically useful
