# Recommendation algorithm paper

> [中文版](2025-01-07-tencent-diffgraph.md)

```
1 Recommendation level: currently only two — "worth a look" and "read if you have time". Occasionally "strongly recommended".
2 * Worth a look: people working on ranking should get to know it promptly; it helps with work.
3 * Read if you have time: as the name says.
4
5 Credibility: offline or production.
6 * Production: methods that can be used in production; relatively more useful, and others are more willing to apply them to their own business.
7 * Offline: by comparison, somewhat less so.
```

1 DiffGraph: Heterogeneous Graph Diffusion Model

Link: https://arxiv.org/pdf/2501.02313

Company: Tencent

Area: GNN + diffusion

Overview: uses the strong denoising ability of diffusion to help denoise during GNN modeling. Instead of adding and removing noise directly on the raw graph structure, the diffusion runs in the graph's latent representation space.

Credibility: offline data

Recommendation: read if you have time
