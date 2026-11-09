# Recommendation Algorithms (2025.6.11): Google's serendipitous recommendation

> [中文版](2025-06-11-google-serendipitous.md)

1 Serendipitous Recommendation with Multimodal LLM

Link: https://arxiv.org/pdf/2506.08283

Company: Google

Area: LLM + i2i

Overview:

Use a large model to predict content that is both relevant and novel, content that pleasantly surprises, to encourage users to explore and improve their experience. That is, the large model predicts a description of the user's serendipitous interest (a cluster), narrowing the recommendation scope.

Specifically:

(1) Use hierarchical tree clustering to split items into interest clusters at different levels; each cluster has a corresponding text description.

(2) Generate the description of the next interest cluster and find the cluster's ID.

Fine-tune a multimodal large model on the platform's most popular videos; the fine-tuned MLLM understands user interests and multimodal content and generates a description of the next serendipitous interest cluster. This description is matched against the nodes (clusters) in the hierarchical clustering tree, and the best-matching node is the corresponding cluster.

High-satisfaction serendipitous (context video, recommended video) pairs are extracted from user logs to fine-tune the MLLM. The MLLM processes video frames, thumbnails and text (such as title and description) to understand the content in depth. A Chain-of-Thought (CoT) prompting strategy is also proposed, using structured reasoning to improve the MLLM's representation of video content.

(3) Low-level recommendation: feed the cluster ID produced by the MLLM into a conventional recommendation model (such as a Transformer-based sequential model) to efficiently retrieve specific items to recommend.

At the lower level, the cluster ID constrains the Transformer-based sequential model in these ways:

* Embedding input: the cluster ID is turned into an embedding vector and added to the user interaction sequence as context or a condition.

* Action space restriction: the Transformer's output layer only considers candidate items within the given cluster, filtering out items from other clusters.

* Efficient retrieval: by precomputing embeddings or indexes of items within a cluster, the model can quickly rank or select the most relevant recommendations.

Takeaways: one way for a large model to assist a recommendation model.

Credibility: production

Recommendation: worth a look
