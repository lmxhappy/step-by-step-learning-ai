# Recommendation papers (2025.5.8): eBay uses a hierarchical GNN + LLM for ad keyword recommendation
> [中文版](2025-05-08-ebay-ad-keyword-gnn-llm.md)

> Source: WeChat post https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484636&idx=1&sn=8ab523471efbb4cd3e03edb68ca79dea&scene=21#wechat_redirect, published 2025-05-08.
> Split from the roundup "Recommendation papers (2025.5.8): great work from eBay, ByteDance and Xiaohongshu"; same issue: [eBay hierarchical GNN + LLM ad keyword recommendation](../deep-retrieval/2025-05-08-ebay-ad-keyword-gnn-llm.full.en.md), [ByteDance S3 self-supervised long-sequence modeling](../behavior-sequence-modeling/2025-05-08-bytedance-s3-long-sequence.full.en.md), [Xiaohongshu action-oriented sequence modeling](../behavior-sequence-modeling/2025-05-08-xiaohongshu-action-oriented-seq.full.en.md).

Enhancing Ad Keyword Recommendation via Hierarchical Graph Neural Network and LLM

Link: https://arxiv.org/pdf/2504.21187

Company: eBay

Area: keyword retrieval

Overview: This paper proposes an ad keyword recommendation method based on a graph neural network (GNN) and a large language model (LLM), to solve the problem that sellers find it hard to pick effective keywords when creating ads. By building a heterogeneous seller-ad-keyword graph and combining it with LLM-generated keyword descriptions, the model learns the semantic relations between keywords, giving efficient and precise keyword recommendations and clearly improving seller satisfaction and ad performance.

A/B: in the online A/B test, keyword recommendation CTR rose 2.3% and seller satisfaction rose 5.1%.

Takeaways:

(1) Combining graph neural networks with large language models is a very promising direction.

(2) Keyword recommendation is a very practical problem and worth studying in depth.

Credibility: production (eBay)

Recommendation: worth a look
