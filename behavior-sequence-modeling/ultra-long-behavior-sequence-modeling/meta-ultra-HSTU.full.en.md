# Meta, HSTU 2.0

> [中文版](meta-ultra-HSTU.md)

### Paper: Bending the Scaling Law Curve in Large-Scale Recommendation Systems
### Link: https://arxiv.org/pdf/2602.16986
### Company: Meta
### Idea: scaling law, sparse attention, low-precision quantization
### Area: ultra-long behavior sequence modeling

## Overview:
This is a method for ultra-long sequence modeling. User features + context features + the user interaction history (UIH) + multiple candidate items are organized together into one long token sequence, then passed through a stack of STU (Sequential Transduction Unit) layers (i.e., self-attention layers) so that all elements in the sequence fully interact and exchange information, finally directly producing multi-task predicted probabilities for each candidate (click, completion, share, watch time, etc.).

All the techniques in this paper serve to increase the modeled sequence length, including:

### (1) Input sequence optimization
* In the sequence, item and action embeddings are no longer interleaved; instead the two are added together, which directly halves the sequence length. Note that the user history sequence is ordered from newest to oldest.
* Heterogeneous action encoding: each action embedding becomes a very information-dense vector, combining "what action + action strength + the user context at that time".
* Load-Balanced Stochastic Length (LBSL): for each user sequence, randomly decide a truncation length k and keep only the first k actions of the sequence (i.e., the user's most recent k interactions).

### (2) Redesigning the self-attention mask
Proposes Semi-Local Attention (SLA): each sequence element only attends to a small set of items, in two parts: nearby local and recent global.
* Nearby local: the sequence elements just before and after it, length K1;
* Recent global: the user's most recent stretch of sequence elements, length K2.
Other items are not attended to; in other words, their attention scores are 0.

### (3) Mixed precision
* In training, matrix multiplications use FP8, and everything else uses BF16.
* In inference, embedding lookup uses int4. Transferring the looked-up embeddings from host memory to GPU HBM also uses int4.

### (4) Custom kernels
Multiple operators, such as quantization, scaling, LayerNorm, matrix multiply and matrix add, are fused into one custom kernel, saving multiple memory reads/writes and improving compute efficiency.

### (5) Reducing GPU memory usage
* Activation recomputation: a standard Transformer saves many intermediate activations in the forward pass for the backward pass, so memory explodes for ultra-long sequences (16k). The 6 largest forward tensors are not saved and are recomputed in backward, which makes it possible to fit ultra-long sequences in GPU memory.
* Jagged Tensor: in traditional training, user sequences of different lengths must be padded to the maximum length (e.g., all padded to 16k), wasting a lot of memory. Jagged tensors are used throughout, so variable-length sequences are truly not padded, and all operations (attention, GEMM, LayerNorm, etc.) support the jagged format.

### (6) Pyramid stacking
Like OneTrans, this paper also does pyramid stacking, with the same purpose as **OneTrans**: later layers don't need to process the ultra-long sequence, saving compute. The difference is that it is not strictly "each layer a bit shorter" but two-stage: the first N1 layers use the full sequence, and the last N2 layers only process the most recent high-value subsequence.

### (7) Sequence splitting
The consumption sequence and the interaction sequence each go through their own HSTU and are then fused, so high-value signals are not diluted. This also reduces compute, so more and older history can be included.

### Beyond this, the paper's insights:
* Self-attention still beats cross-attention, especially in stackable depth and scaling.
* "Model-system co-design" is the real key to landing in industry (a recommendation version of the DeepSeek-V2 approach). For the first time, self-attention is truly scalable on industrial-scale 16k+ ultra-long user sequences.

**A/B**: 30 days, hundreds of millions of users:
* Consumption metrics (watch time, completion, etc.): +4.11%
* Interaction metrics (likes, comments, shares, etc.): +2%–8.2%
* Topline metrics (DAU, visits and other confidential metrics): +0.217% / +0.037% (considered a major breakthrough inside Meta). This is one of the biggest model upgrades on Meta's recommendation platform in recent years.

## Takeaways:
* This paper is the product of the algorithm team and the ML systems team working together.
* Key tricks from some Chinese companies' work, such as OneTrans, DeepSeek and STCA, are being studied and referenced by big US companies.
* Meta has the PyTorch team, which makes custom kernels easier.
* The paper uses both NVIDIA and AMD GPUs, mixing multi-vendor GPUs to reduce supply-chain risk and optimize cost.

## Caveats
The history sequence is ordered from newest to oldest, but the pyramid in Figure 2 makes it look as if the old part is kept.

## Credibility: production

## Recommendation: practically useful

![pic](../../imgs/ultra-hstu.png)
