# Recommendation algorithm papers

> [中文版](2024-12-25-huawei-graph-construction.md)

1 An Automatic Graph Construction Framework based on Large Language Models for Recommendation

Link: https://arxiv.org/pdf/2412.18241

Company: Huawei

Area: GNN

Overview: proposes an LLM-based automatic graph construction framework for recommendation, with two stages: quantization-based graph construction and graph-enhanced recommendation.

(1) In the quantization-based graph construction stage, LLMs are first used to infer user preferences and item knowledge, which are encoded into semantic vectors. Using point-wise invocation (i.e. calling the LLM separately for each user or item), latent factors are extracted for users and items with vector quantization. By adding the latent factors as extra nodes, the new graph contains user nodes, item nodes and latent-factor nodes. This builds a graph with a global view and deep semantics, giving more comprehensive and informative insight from the topology.

(2) In the graph-enhanced recommendation stage, a meta-path-based message propagation method is proposed: a graph attention network (GAT) on the constructed graph aggregates information from different meta-paths, producing enhanced user and item representations.

This generates graph-enhanced user/item representations, which can be seamlessly integrated into any recommender system to improve performance.

Credibility: production. Metrics up substantially.

Recommendation: worth a look
