# Recommendation papers (2025.1.3): Kuaishou does collaborative filtering with a diffusion model in the spectral domain
> [中文版](2025-01-03-kuaishou-s-diff.md)

> Source: WeChat post https://mp.weixin.qq.com/s/kx6LdmgPOSOOGQYsyiME_Q, published 2025-01-03.
> Split from the roundup "Recommendation papers (2025.1.3)"; same issue (2 papers): [Kuaishou S-Diff spectral-domain diffusion CF](../graph-recommendation/2025-01-03-kuaishou-s-diff.full.en.md), [Reuters NewsHomepages news prioritization](../loss-func/2025-01-03-reuters-newshomepages.full.en.md).

S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain

Link: https://arxiv.org/pdf/2501.00384

Company: Kuaishou

Area: Collaborative filtering + diffusion

Overview: Applies a diffusion model to collaborative filtering in the spectral domain. The model first maps user interaction vectors into the spectral domain and uses the graph's frequency information to parameterize the diffusion noise, preserving low-frequency components to maintain a high signal-to-noise ratio. It then uses a conditional denoising network to encode user interaction information and recover the true preferences from the noisy data.

Credibility: Offline

Recommendation: Read if you have time
