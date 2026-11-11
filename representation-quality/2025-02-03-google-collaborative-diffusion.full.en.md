# Recommendation papers (2025.2.3): Google Collaborative Diffusion Model: pseudo-users from reviews to enrich user representations
> [中文版](2025-02-03-google-collaborative-diffusion.md)

> Source: WeChat post https://mp.weixin.qq.com/s/qvI9Ac5AJ96QwL3DoSPP5Q, published 2025-02-03.
> Split from the roundup "Recommendation algorithm papers (2025.2.3)"; same issue: [Amazon CHARM multi-field retrieval](../deep-retrieval/2025-02-03-amazon-charm.full.en.md), [Google collaborative diffusion](../representation-quality/2025-02-03-google-collaborative-diffusion.full.en.md).

Collaborative Diffusion Model for Recommender System

Link: https://arxiv.org/pdf/2501.18997

Company: Google

Area: diffusion-based recommendation

Overview: Uses diffusion to bring in review data and improve user representations. Three main steps: (1) Pseudo-user generation: each feature (word) in the item reviews is treated as a pseudo-user; feature importance is computed with TF-IDF and normalized to build pseudo-user interaction vectors. This fits into the existing CF framework and also leverages the denoising ability of DR, making item-side information and user signals complementary and improving recommendation quality. (2) Personalized neighbor identification: based on behavioral similarity, for each query user the top neighbors are identified separately from real users and from pseudo-users, using cosine distance to determine the neighbor sets; these neighbor sets are computed and cached during preprocessing so training and inference can use them efficiently. (3) Collaborative signal aggregation: during the diffusion process, the preference information (PI) of real and pseudo neighbors is aggregated to obtain a finer-grained user preference representation. Neighbor attention scores are computed in three ways: average pooling, behavioral similarity, and parametric modeling, giving the model flexible options. The objective function is reformulated accordingly to optimize the model.

Credibility: offline

Recommendation: read if you have time
