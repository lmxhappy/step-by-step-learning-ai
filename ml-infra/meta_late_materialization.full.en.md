# Meta: the recommendation team's engineering architecture is impressive

> [中文版](meta_late_materialization.md)

### Paper: Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale
### Link: https://arxiv.org/pdf/2604.24806
### Company: Meta
### Idea: trading compute for storage
### Area: recommender system infrastructure

## Overview:
This paper changes "store samples at inference time for later training" into "store version information at inference time, and at training time rebuild samples on demand from a store using that version information".

### Problem background
In deep learning recommender systems, inference and training must stay consistent. At inference time, the full features of the user's complete interaction history, plus other features, are saved in advance as training samples. This creates a core problem: **when sequence length grows from a few hundred to several thousand, data infrastructure resource consumption far exceeds what GPU training needs**. On top of that, different tenants store the same data repeatedly, or store sequence data of different lengths.

### Core innovations:
* Storage: build an immutable storage layer that stores history in segments by start timestamp, supporting efficient range scans. Each user's history is stored once, and multiple tenants access the same data in parallel with no duplicate storage.
* Inference: only record the mutable part (the most recent real-time events) + a version pointer to the immutable part (metadata including start_ts, end_ts, length, checksum, etc.).
* Training: use the version pointer + a time predicate (timestamp ≤ request_time) to rebuild the full sequence on the fly, which naturally prevents future leakage.

This rebuild logic can serve both real-time training and offline training.

When you hear that training can't use the samples cached at inference and instead rebuilds them, you might wonder whether this adds compute and slows training. This is solved: the techniques below fully hide the rebuild latency, and training throughput is still dominated by GPU compute.
* Hybrid storage: a mutable layer (real-time KV) + an immutable layer (compacted daily into a columnar, read-optimized format, supporting multi-dimensional projection pushdown).
* Read-optimized immutable layer: columnar encoding + projection pushdown, so each tenant's model reads only the columns and length it needs, eliminating the multi-tenant penalty.
* Training-time optimizations: decoupled preprocessing, pipelined I/O prefetch, data-affinity scheduling.

## Takeaways:
* This is **a model example of system design and engineering practice**. Not pure algorithmic innovation, but clever architecture design solving an infrastructure dilemma at production scale.
* The idea of "late materialization" is well worth borrowing: **don't do unnecessary precomputation on the critical path**; defer it until it's actually needed, similar to lazy evaluation in databases.
* The versioning mechanism solves the classic online-offline consistency problem, and especially in a multi-tenant environment its value is huge.
* The complexity of this kind of infrastructure work is often underestimated, but its business impact can be larger than a single algorithm improvement: supporting longer sequences = better model quality.

## Credibility: production

## Recommendation: practically useful

![pic](../imgs/meta_late_materialization.png)
