# Recommendation Algorithms (2026.1.5): Tencent, one-stop recommendation, watch time +1.22%

> [中文版](2026-01-04-tencent-higr.md)

1 HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment

Link: https://arxiv.org/pdf/2512.24787

Company: Tencent

Idea: generative recommendation that generates the whole recommendation list in one pass

Area: generative recommendation + listwise + one-stop recommendation

Overview:

It proposes a generative model that generates the whole recommendation list, made up mainly of SID generation, the generative model, and post-training.

(1) Generating SIDs: Contrastive RQ-VAE (CRQ-VAE)

To address the following problems:

1) The tokens of SIDs from RQ-VAE have unclear semantics; there is a semantic gap;

2) In RQ-VAE, the same item may be mapped to different SIDs (lack of uniqueness);

3) In RQ-VAE, different items may be mapped to the same SID (lack of distinctiveness).

So it proposes Contrastive RQ-VAE (CRQ-VAE), which mainly includes:

1) Reconstruction loss: keeps the original RQ-VAE reconstruction loss to preserve the semantic information of SIDs;

2) Contrastive loss: through contrastive learning, pulls SIDs of the same item closer and pushes SIDs of different items apart, improving the uniqueness and distinctiveness of SIDs;

3) Diversity loss: encourages differences between SIDs of different items, improving SID diversity.

(2) The generative model

By stacking three transformations, it gradually turns the user behavior sequence into the final recommendation list.

1) Context

The SIDs of the items in the user behavior sequence and the item features (e.g. category, price, brand) are concatenated into a vector and passed through an MLP to get a vector, used as the context representation.

2) Coarse-grained layer

The context representation is passed through a transformer layer to get a coarse-grained representation of the recommendation list.

And these transformation layers are stacked.

Note that each transformation has a ResNet-style residual (skip) connection.

3) Fine-grained layer

The output of the coarse-grained layer is passed through a transformer layer to get a fine-grained representation of the recommendation list.

NTP loss is computed on the predictions to optimize the model.

(3) Post-training

Supervised contrastive learning is used for post-training to align with user preferences. It's binary classification over constructed preference pairs: positives are the user's real interaction sequences (reordered by feedback in descending order); negatives include random permutations (to optimize ranking), negative-feedback replacement (to optimize interest), and repeated-anchor appending (to optimize diversity). This ensures generation jointly optimizes ranking, interest and diversity.

(4) Inference

The generative model autoregressively predicts M items, using beam search during prediction. Items are determined by reverse lookup of the item corresponding to each codebook entry.

A/B: on WeChat Channels, with 2% traffic: average dwell time +1.03%, watch time +1.22%, video views +1.73%, requests +1.57%.

Supplement:

Problems with RQ-VAE:

1) ID entanglement and sparsity: RQ-VAE's SID tokens are semantically entangled; SID tokens of different items may overlap, making it hard for the model to tell items apart.

2) Lack of collaborative alignment: RQ-VAE's SID tokens are not collaboratively aligned with user behavior, making it hard for the model to learn the relationship between user behavior and items.

3) Inefficient diversity control: RQ-VAE's SID tokens lack diversity control, so the generated recommendation list may contain repeated or similar items.

Takeaways:

(1) Listwise approaches have rarely been implemented successfully, and since generative recommendation can output multiple items, it can serve as one way to implement listwise.

Caveats:

Formula (10) has a typo.

Credibility: production

Recommendation: practically useful
