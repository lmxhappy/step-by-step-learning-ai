# Alibaba, a new feature crossing method, CTR +3%

> [中文版](ali-mgdin.md)

### Paper: Deferred is Better: A Framework for Multi-Granularity Deferred Interaction of Heterogeneous Features
### Link: https://arxiv.org/pdf/2603.12586
### Company: Alibaba Lazada
### Idea: deferred interaction
### Area: feature crossing

## Overview:
This paper proposes a new feature crossing method.
The problem with feeding all features into the interaction layer at once is that low-information sparse features bring a lot of noise early on and drown the signal of high-quality features. Put the other way, some added high-quality features get squeezed out as the network propagates forward, especially during feature crossing.
This paper's method addresses exactly that. In short: features of different information density should not join the crossing at the same moment; sparse features should be brought in later.

### (1) Feature grouping
Features are sorted by information content from high to low, then grouped under several grouping schemes. Each grouping scheme goes into the later interaction module **independently and in parallel**.
For one grouping scheme, say n groups, each group is concatenated into a group representation, giving a vector sequence of length n.

**Note**:
The paper does not explain how information content is defined. There are many possible ways to rank: frequency counts, embedding norm, feature cardinality, information gain, and so on. Each makes sense, but the paper mentions none of them. The chain of logic is: the sparser a feature (say, an unpopular item ID), the fewer times it appears in the training data, the fewer times its embedding gets trained, so the learned representation is worse and noisier. Conversely, low-cardinality or numeric features such as device type or user age appear in almost every sample, their embeddings are well trained, and the representation quality is high.

### (2) Feature interaction
Each grouping scheme does feature interaction between its groups, essentially Transformer self-attention, producing one representation. Finally, the representations of all grouping schemes are concatenated and passed through an MLP for prediction, with cross-entropy loss.

For one grouping scheme, the length-n vector sequence goes through self-attention to get a sequence of the same dimension. Then come a residual connection, a LayerNorm layer, a feed-forward network, a LayerNorm layer, and so on. Stacking several such blocks gives one representation. So the difference from the Transformer is that there is no softmax (and related operations) on the attention.
The concrete meaning of the residual connection here is that it carries this feature group forward, so it can be crossed with other feature groups in later layers. During interaction, each feature group keeps absorbing information from the other groups and strengthening itself, achieving "lossless" forward propagation.

In particular, when computing self-attention, a top-K mask is added: only the top-K largest entries of the whole attention matrix are kept, and the other attention values are set to 0.
Note:
* The K of top-K differs per layer, growing layer by layer according to a formula.
* The mask is computed from the first layer's attention matrix and reused by all later layers. **Why not recompute it at each layer?** Because as the network deepens, each group's representation has already absorbed information from other groups, so the scores no longer reflect the "original information content". The first layer's representations are the purest, and its scores best show which groups are information-rich (high score) and which are sparse and noisy (low score).

**A/B**: Lazada, +3.04% relative CTR.

**Why do it this way?**
Two angles:
* Avoid noise propagation. Shallow layers are the key stage where representations are built. If sparse features join the interaction at this stage, their under-trained embeddings spread noise into the representations of dense features, a case of "bad money driving out good". The mask cuts this propagation path.
* Stabilize first, then expand. Dense features first interact fully in the shallow layers, building a stable, high-quality representation base. By the deeper layers this base is robust enough that, when the noise of sparse features comes in, the model has enough "immunity" to absorb the useful signal and filter the noise.

## Takeaways:
* Before large models arrived, feature crossing was never perfectly solved. Now that large models are here, fewer people work on it and there is less interest, but it still needs solving, because solving it still yields results.

## Credibility: production

## Recommendation: practically useful

![pic](../imgs/MGDIN.png)
