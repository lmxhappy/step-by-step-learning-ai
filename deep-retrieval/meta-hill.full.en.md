# Meta: tree-structured clustering index + deep retrieval model, business metric +2.57%

> [中文版](meta-hill.md)

### Paper: Efficient Retrieval Scaling with Hierarchical Indexing for Large Scale Recommendation
### Link: https://arxiv.org/html/2604.12965
### Company: Meta
### Idea: clustering, residual quantization
### Area: deep retrieval

## Overview:
This paper builds an end-to-end trainable tree-structured clustering index, trained jointly with a deep retrieval model. Because the item corpus can run into the hundreds of millions, an index is necessary to cut computation, which is what makes it possible to put a deep retrieval model into production. Put simply, you can picture it like this: the leaf nodes of the tree are real items, and each non-leaf node is a representative item of all its leaf descendants. With this picture, you already understand most of the method. During training, index nodes are treated just like real items; during inference, the top nodes are picked among the non-leaf nodes of the same level, which narrows the retrieval range, reducing computation and speeding up retrieval.

Take a 3-level tree index as an example:
Level 1 (root): a small number of virtual "coarse cluster centers" (index nodes)
Level 2: medium-granularity virtual nodes, learned "virtual cluster centers".
Level 3 (leaves): real items (embeddings precomputed by the MoNN Item Tower)

As you can see, the lower the level, the more nodes there are, representing clustering and grouping from coarse to fine. All nodes (both index nodes and leaf nodes) are trained end to end together with MoNN (joint training). How MoNN learns the leaf nodes is easy to understand and not covered here. How index nodes are learned is more particular:

### (1) Learning the index nodes
Use the item embedding as the query and the learned index-node embeddings as key/value, do cross-attention, and get a pseudo-representation.
Feed the pseudo-representation to MoNN as a "virtual item", output a preference logit, and supervise it with the hard label y of the corresponding real item. The gradient flows back through the pseudo-representation to the index-node embeddings, thereby optimizing the index nodes indirectly.
Upper-level index nodes use a large MoNN network; lower levels use a medium or small MoNN model.

In particular, the virtual features of the index nodes used for the coarse-level I2IF are computed as follows:
* (1) Delayed, batched, non-real-time: during the earlier cross-attention computation, the items most similar to an index node form a set, and the corresponding attention values are recorded.
* (2) For a one-hot feature, pick the feature value with the highest weighted_count and look up its embedding in the embedding layer; for a multi-hot feature, look up the embeddings of feature values with weighted_count greater than 0 and pool them: a cluster center.
    * How a feature value's weighted_count is computed: within the set of real items corresponding to the index node, take a weighted sum over all items that have that feature value, weighted by the attention values, giving a count weighted_count.

As in the MoNN network described in [HSNN](./meta-hsnn.full.en.md), real items are trained not only with multi-task supervised learning on labeled data, but also with semi-supervised learning on more unlabeled data to improve the model. This paper trains index nodes in a similar way, like real items.

### (2) Residual Quantization
The previous level's residual minus the pseudo-representation is this level's residual. In particular, the first level's residual is the item embedding.
So the levels pass residuals between them: upper levels capture coarse semantics, lower levels capture fine details, and no information is lost.

For a given real item, there is an auxiliary task on the sum of its pseudo-representations over all index levels versus its embedding: a reconstruction loss is introduced, forcing the hierarchical index to preserve as much of the original item's semantics as possible through "compression + reconstruction".

### (3) Test-Time Training (TTT)
The trained intermediate index nodes naturally correspond to high-quality machine-readable categories. Treat these intermediate index nodes as new "pseudo-items", generate <user, index node> training pairs (with simple filtering), and do label-free fine-tuning of MoNN at inference time (test-time training). The user's current inference is not affected; the updated MoNN parameters are only used for subsequent scoring within the current user request's session. The user's later sessions are not affected and use the global MoNN model, and the model used for other users is not affected either.

Training uses tricks such as a temperature scheduler and FLOPs regularization (to prevent cluster collapse) to make index learning more stable.

### Inference
At inference, beam search walks quickly down from the root, expanding only a small number of the best paths, and in the end the full MoNN + OverArch is run on only a few thousand real items.
Concretely, retrieval starts at the root (the coarsest index nodes), scores quickly with MoNN Large, and traverses level by level from top to bottom.

After each level, only the top-k best index nodes are kept as candidates for the next level. In the end only very few paths reach the leaf nodes (real items), and only those items get the full OverArch scoring.

### A/B:
Meta ads: MoNN with a 2-level index vs. the base MoNN improved online business metrics by up to +2.57%. It already supports daily ad recommendation for hundreds of millions of users on Facebook + Instagram.

## Takeaways:
* HILL's hierarchical index is essentially the RQ-VAE idea of "multi-level residual cluster centers" transplanted onto a learnable index for recommendation, achieving "fast filtering at coarse levels + precise scoring at fine levels".
* In GR (generative recommendation), a tree is built to speed up decoding; this paper also does a similar clustering into groups, but the difference is that here the parent-descendant relations are dynamic, not fixed in either inference or training.

## Credibility: production

## Recommendation: has practical value

![pic](../imgs/hill.png)
