# Daily pick (2025.5.12): a small discovery from Pinterest

> [中文版](2025-05-12-pinterest-embedding-table.md)

1 The Evolution of Embedding Table Optimization and Multi-Epoch Training in Pinterest Ads Conversion

Link: https://arxiv.org/pdf/2505.05605

Company: Pinterest

Area: embedding table optimization

Overview: The paper proposes a way to optimize the embedding layer of a recommendation model: give it a learning rate different from the main network's. It consists of two small methods: raise that layer's learning rate, but lower the learning rate for its low-frequency feature values. The two are combined; the first applies to all features, the second to some of them. In other words, low-frequency feature values get the combined effect of both. Specifically:

(1) Raise the layer's learning rate: set a higher layer-specific learning rate for the embedding tables, based on the Adam optimizer, while keeping the learning rate of the other dense parameters unchanged.

(2) Frequency-adaptive learning rate: lower the learning rate for rarely updated rows in the embedding tables (low-frequency IDs), to ease overfitting in multi-epoch training while keeping performance on high-frequency IDs. The learning rate is scaled by the row's relative log frequency (log scaling), which works better than linear scaling.

Why are low- and mid-frequency rows, with fewer updates, more prone to overfitting?

Because low-frequency rows are updated from few samples, and gradient updates based on very few samples have high variance, their embedding vectors easily "memorize" the specific patterns or noise of those samples instead of a generalizable feature representation. For example, a low-frequency ad ID may appear in only a few positive samples, and the model may over-optimize its embedding to fit them, biasing predictions on its future samples. Conversely, rows that are trained often (high-frequency rows) are less likely to overfit, mainly because:

* Low-variance updates: high-frequency rows are updated from many samples, gradient estimates are stable, and embeddings tend toward generalizable representations. Statistical stability: the embeddings of high-frequency rows are updated from many samples, consistent with the law of large numbers, so gradient estimates are closer to the true distribution, reducing overfitting risk.
* Diverse contexts: high-frequency rows cover many scenarios, which strengthens generalization.
* Robustness to sparse labels: high-frequency rows are associated with more positive samples, reducing overfitting to a few samples.
* Safe with high learning rates: high-frequency rows can tolerate high learning rates (e.g. a Sparse Optimizer), converging faster without easily overfitting.

Credibility: production

Recommendation: practically useful (worth a try)
