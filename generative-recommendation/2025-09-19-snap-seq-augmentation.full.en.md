# Recommendation Algorithms (2025.9.19): Snap's data augmentation for sequence samples, practically useful

> [中文版](2025-09-19-snap-seq-augmentation.md)

1 Sequential Data Augmentation for Generative Recommendation

Link: https://arxiv.org/pdf/2509.13648

Company: Snap

Area: behavior sequence + generative recommendation + data augmentation

Overview:

Data augmentation for the training samples of generative recommendation models, to address sample sparsity, especially for long-tail samples. That is, by sampling a subsequence and a target item, a sample is generated for training the generative recommendation model.

Sampling process:

(1) Sequence sampling: sample a sequence u from all user sequences. This samples a user.

(2) Target sampling: in the chosen sequence, sample a position k as the next item (target y).

(3) Input sampling: sample a subsequence before the target item as the input x. This samples a subsequence from a complete sequence.

The three sampling steps (sequence sampling, target sampling, input sampling) are combined in order to produce training pairs (behavior sequence, target item) for training the generative recommendation model.

Background — strategies for building generative recommendation training samples, simplified into three:

(1) Last-Target (LT): only one pair per user: the input is the sequence minus the last item, and the target is the last item. Simple, but ignores the supervision signal inside the sequence.

(2) Multi-Target (MT): one pair for each position k≥2: the input is the first k-1 items, and the target is the k-th item. Provides more signal, but the input is always the full prefix.

(3) Slide-Window (SW): more aggressive; for each target k, enumerate input subsequences of every possible window size. High diversity, but computationally expensive and may introduce noise.

Takeaways:

(1) Data augmentation is a common technique. When a new direction appears, consider applying it to get an innovation.

Caveats: behavior sequence modeling could also try this data augmentation method.

Credibility: production

Recommendation: practically useful
