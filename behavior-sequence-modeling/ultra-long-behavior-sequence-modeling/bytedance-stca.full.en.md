# Recommendation papers (2026.2.26): ByteDance, ultra-long sequence modeling, completion rate +3%

> [中文版](bytedance-stca.md)

### Paper: Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin
### Link: https://arxiv.org/pdf/2511.06077
### Company: ByteDance
### Idea: stacking
### Area: ultra-long behavior sequence modeling

## Overview:
The paper proposes a method for long behavior sequence modeling. It drops self-attention and uses only target attention, making the network bigger through so-called stacking to improve modeling quality. The resulting user representation is fed into the main network to predict the probabilities of the user's various actions.

### (1) STCA: stacked target attention
To make the model bigger, multiple target attention layers are stacked, in the hope of getting scaling-law-style gains.

This is not traditional stacking where one layer's output is the next layer's input. Instead, the user interest representation obtained from a target attention layer is combined with the target item's embedding to produce a new query, which is used for this layer's attention computation.

I drew a simple diagram. The bottom left is the behavior sequence; the green on the right is the candidate item (used as the query). A layer's query is the concatenation of the user interest representations computed by the previous layers' attention and the original candidate item representation, passed through a simple network. To keep it simple, only the data sources of layer 3's query are marked.

STCA's final output representation is the concatenation of all layers' user interest representations.

![alt text](../../imgs/stca.png)

### (2) Request-level batching
This is a training-time trick. It is essentially the multi-candidate prediction from HSTU. A sample has only one candidate item, but one user request has multiple samples, all with the same user behavior sequence.

This paper puts multiple candidates of the same request (note: not all of them) into one batch, which reduces data transfer and also cuts some compute.

### (3) Random sampling of the behavior sequence
HSTU was the first to do this. The purpose and benefit is saving transfer and compute cost. Quality definitely takes some hit; the hope is to trade a tiny quality drop for a huge compute reduction, so that long sequences can make it into production.

That is, the behavior sequence is randomly sampled during training, and the full 10k sequence is used at inference. The paper gives it an elegant name: Train Sparsely, Infer Densely. Here sparse / dense does not refer to model parameters but to the sequence length (number of tokens) actually processed per training/inference sample.

They find it keeps 80% of the gain of full-10k training while cutting compute by 2/3. Clearly well worth it.

**A/B**: over one month, completion rate rose substantially by 3.3%–4.2%, while core metrics such as app time spent, comments and likes were all positive; low- and mid-activity users gained the most.

## Takeaways:
* After target attention you get a single representation, so with traditional stacking there is no way to stack: you can't do anything on the key and value side, and can only stack on the query. That is why nobody did stacking, and the stacking method this paper came up with is quite special.
* For long behavior sequence modeling, a single layer of target attention doesn't work, or the effect is not noticeable; you have to find a way to stack.

## Caveats
Self-attention can't be dropped completely; it should still be kept in some form, e.g. handling the short sequence in the traditional way (self-attention + target attention). It's understandable that the paper doesn't say so, since it wants to showcase its own method.

## Credibility: production

## Recommendation: practically useful

![pic](../../imgs/stca.png)
