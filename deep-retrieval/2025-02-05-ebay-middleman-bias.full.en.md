# Recommendation papers (2025.2.4): eBay middleman bias: aligning ad keyphrase recommendation relevance with search
> [中文版](2025-02-05-ebay-middleman-bias.md)

> Source: WeChat post https://mp.weixin.qq.com/s/0-a1tBUDlUzowfI-MuDTBA, published 2025-02-05.
> Split from the roundup "Recommendation algorithm papers (2025.2.4)"; same issue: [Alibaba MIM multimodal interest modeling](../behavior-sequence-modeling/2025-02-05-ali-mim-multimodal-interest.full.en.md), [eBay ad keyphrase middleman bias](../deep-retrieval/2025-02-05-ebay-middleman-bias.full.en.md).

Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search

Link: https://arxiv.org/pdf/2502.00131

Company: eBay

Area: ad search + keyphrase recommendation

Overview: This paper studies the relevance of ad keyphrase recommendations on the eBay e-commerce platform. It finds that training a relevance filtering model on click/sales signals is flawed, because the data is biased and search ranking affects click behavior. The paper proposes the concept of "middleman bias" and stresses that ad keyphrases need to be consistent with search relevance signals. It compares bi-encoders and cross-encoders on how well and how scalably they model this consistency. The results show that a simple two-layer cross-encoder does better at relevance modeling; although bi-encoders are fast, a lightweight cross-encoder can close the latency gap and give better results.

Credibility: production

Recommendation: read if you have time
