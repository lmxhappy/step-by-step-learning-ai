# Recommendation papers (2025.1.3): Reuters learns news importance ranking from homepage layouts
> [中文版](2025-01-03-reuters-newshomepages.md)

> Source: WeChat post https://mp.weixin.qq.com/s/kx6LdmgPOSOOGQYsyiME_Q, published 2025-01-03.
> Split from the roundup "Recommendation papers (2025.1.3)"; same issue (2 papers): [Kuaishou S-Diff spectral-domain diffusion CF](../graph-recommendation/2025-01-03-kuaishou-s-diff.full.en.md), [Reuters NewsHomepages news prioritization](../loss-func/2025-01-03-reuters-newshomepages.full.en.md).

NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions

Link: https://arxiv.org/pdf/2501.00004

Company: Reuters

Area: News importance judgment + pair-wise

Overview:

They collected snapshots of the homepages of more than 3,000 news websites, built pairwise samples based on each article's position and size on the homepage, and then used a Transformer-based binary classifier to predict the relative importance of articles. On top of that, there are two downstream tasks: one analyzes the similarity of news-value judgments across different news organizations; the other assesses the newsworthiness of news. You can see how seriously and diligently Reuters monitors news developments in every field around the world, and it also shows where their news authority comes from: technological innovation.

Credibility: Production

Recommendation: Read if you have time
