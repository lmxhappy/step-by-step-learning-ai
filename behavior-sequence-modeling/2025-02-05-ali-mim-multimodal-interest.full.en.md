# Recommendation papers (2025.2.4): Alibaba MIM: multimodal content interest modeling for user behavior sequences
> [中文版](2025-02-05-ali-mim-multimodal-interest.md)

> Source: WeChat post https://mp.weixin.qq.com/s/0-a1tBUDlUzowfI-MuDTBA, published 2025-02-05.
> Split from the roundup "Recommendation algorithm papers (2025.2.4)"; same issue: [Alibaba MIM multimodal interest modeling](../behavior-sequence-modeling/2025-02-05-ali-mim-multimodal-interest.full.en.md), [eBay ad keyphrase middleman bias](../deep-retrieval/2025-02-05-ebay-middleman-bias.full.en.md).

MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling

Link: https://arxiv.org/pdf/2502.00321

Company: Alibaba

Area: multimodal + behavior sequences

Overview: Uses multimodality for behavior sequence modeling. Before a multimodal model is used for behavior sequence modeling, it needs pre-training and alignment on your own business data: take a good public pre-trained model (CV, NLP, or both), pre-train it further on your own business, and align image with text, query with candidate, and so on.

Several works have already shown this: before the multimodal embedding of a candidate item goes into the recommendation model, it needs thorough pre-training and various alignments, especially alignment with the ID embedding. Stuffing it in directly is not recommended.

Credibility: production

Recommendation: worth reading
