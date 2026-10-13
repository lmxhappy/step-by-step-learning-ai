YouTube stopped logging content embeddings. It logs Semantic IDs and rebuilds the embedding on the accelerator.

Paper: Tokens are All You Need - Dual-purpose Semantic IDs, Google (https://arxiv.org/abs/2607.24865), RecSys 2026 industry-track best paper nomination.

The lineage matters, or this reads as "they stored an integer." Content embeddings used to go straight in as side features, back when sequences were short. Then sequences grew to hundreds of positions, each carrying a dense vector: at length 200 and dimension 256 that is 51,200 floats per example, 200KB in FP32, and across billions of examples it becomes the bandwidth ceiling. Content features got demoted or cut.

Then Semantic IDs solved identity generalization. RQ-VAE quantizes the content embedding into hierarchical tokens, which memorize like atomic IDs and generalize like raw content. SIDs became the standard identity feature for sequence items - but carry only identity, not the quantized-away content.

This paper's insight is almost tautological: the SID was quantized from the content embedding, so it already is a compressed code for it. Production has paid the I/O for SIDs; don't pay twice.

Dual-purpose

The same token sequence is used twice, through two independent paths with independent objectives.

As collaborative identity: each token is a categorical feature into a learnable embedding table, trained by the main recommendation loss - who likes this video.

As content: the same tokens hit the injected codebook, then a trainable lightweight decoder (SiDec, an MLP or shallow transformer) rebuilds the content vector under an MSE loss against the original - what the video is about.

Clean division of labor: the first memorizes, the second generalizes. A new video has no interaction history, so its row in the collaborative table is still at initialization, while the content path works from upload. Cold start rides on it.

Note the decoder is trainable, not frozen. That is the actual contribution - logging SIDs instead of dense vectors was already the baseline.

Results

Throughput is the real story. On retrieval: 16.80 steps/sec with no content feature, 12.07 after adding raw embeddings, 15.41 with SiDec - almost free, and Hit Rate@100 still improves (0.2811, 0.2844, 0.2910).

Deployed across YouTube ranking and retrieval: sitewide satisfied engagement +0.09%, watchpage +0.80%, concentrated disproportionately in new accounts and long-tail content.

Takeaways

- Cost reduction first, quality second. Same "SID as feature" family, different yardstick: not "how much AUC does this feature add" but "how cheap is this feature." If quality holds and cost drops hard, that ships.
- The long-tail gains come from both paths: content needs no interaction history, and nested n-gram prefixes let rare items borrow parameters from frequent ones. Both beat "train the rare IDs longer," which cannot work when the samples aren't there.

#RecSys #RecommenderSystems #MachineLearning #MLOps #AI
