# Recommendation papers

> [中文版](2025-01-22-baidu-diffusion.md)

1 Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation

Link: https://arxiv.org/pdf/2501.11671

Company: Baidu

Area: cross-domain recommendation + diffusion + user cold start

Overview: mainly studies using a preference-guided diffusion model to solve cold start in cross-domain recommendation (CDR). It applies diffusion's forward noising and backward denoising to target-domain user embeddings to improve user embedding quality. In the backward denoising process, the source-domain interaction history is used as the guidance signal, i.e., as input. Clearly, using diffusion to improve embedding quality and solve user cold start is becoming more and more common. For example, the recent DiffGraph: Heterogeneous Graph Diffusion Model (link: https://arxiv.org/pdf/2501.02313) uses diffusion's strong denoising ability to help denoise during GNN modeling.

Credibility: offline

Recommendation: worth a look if you have time.
