# Recommendation Algorithm Papers

> [中文版](2025-02-06-fuxi-huawei.md)

1 FuXi-𝛼: Scaling Recommendation Model with Feature Interaction Enhanced Transformer

Link: https://arxiv.org/pdf/2502.03036

Company: Huawei

Area: behavior sequence

Overview: The paper argues that when self-attention does explicit feature interaction between items, it often falls short at integrating time and position information: it either adds this information to the attention weights or mixes it into the latent representations, which limits the model's expressiveness.

The paper introduces an adaptive multi-channel self-attention mechanism that models time, position and semantic features independently, paired with a multi-stage feed-forward network to strengthen implicit feature interaction. So decoupling time and position helps behavior sequence modeling.

Credibility: production (Huawei Music)

Recommendation: recommended reading.
