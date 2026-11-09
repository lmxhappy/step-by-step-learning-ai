# Recommendation algorithm papers

> [中文版](2025-01-16-bytedance-realtime-indexing.md)

1 Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever

Link: https://arxiv.org/pdf/2501.08695

Company: ByteDance

Area: deep retrieval

Overview: instead of ANN-based retrieval, it achieves real-time deep retrieval in two steps: (1) real-time k-means clustering and assignment, and (2) a DNN that ranks all items in the cluster closest to the item.

Specifically:

(1) Index (on the surface it builds an index; in practice it is real-time k-means clustering and assignment)

Using a two-tower model, run real-time k-means over all items. Compute the cluster centroids of all items in real time, compute each item's embedding in real time, assign each item to its nearest centroid in real time, and update item embeddings and centroid embeddings with gradient descent, which is general, easy to get, and platform-independent. At the same time, record which centroid each item belongs to. At serving time, all items under a given centroid can then be retrieved.

(2) DNN for ranking

During training, this runs in parallel with the part above. At serving time, (1) retrieves the items in the nearest cluster, which are already few in number. So here you can boldly use a DNN for ranking.

If you have read the paper "Enhancing User Interest based on Stream Clustering and Memory Networks in Large-Scale Recommender Systems" (https://arxiv.org/abs/2405.13238), you will understand this paper more deeply. Great minds may simply think alike. The idea of this paper is fairly simple and easy to understand, should give a sizable boost for cold-start items, and is worth trying.

Credibility: production

Recommendation: recommended
