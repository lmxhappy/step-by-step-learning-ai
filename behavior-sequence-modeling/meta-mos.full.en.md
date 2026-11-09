# Meta: sequence clustering + MoE for long-sequence recommendation

> [中文版](meta-mos.md)

### Paper: Mixture of Sequence: Theme-Aware Mixture-of-Experts for Long-Sequence Recommendation
### Link: https://arxiv.org/pdf/2604.20858
### Company: Meta
### Idea: clustering
### Area: long-sequence modeling

## Overview:
The paper proposes a plug-and-play module for long-sequence modeling. It can be integrated seamlessly into any sequential recommendation network, and can also be added to any other network.

### Motivation
The key starting point is the observation of a "Session Hopping" phenomenon: user interests in a long sequence have three properties: stability (interests are stable over the short term), discontinuity (they jump sharply across sessions), and recurrence (past interests may come back).

Implementation:
### (1) Theme-aware routing
Goal: adaptively split the raw long sequence into several subsequences, where items in each subsequence should be very similar (e.g., belong to the same topic), and assign each subsequence to its own expert network; each expert only handles one subsequence.
Computing routing scores: maintain a learnable theme codebook, initialized with k-means. Each sequence item (its embedding) is projected into the theme space with an MLP, and routing scores are computed as cosine similarity against the codebook entries. In practice it is not top-1: sparse activation (top-k expert selection) activates the few most relevant experts. The codebook is continuously updated with an exponential moving average to keep theme centers stable.

All sequence items are assigned to different theme experts. Each theme expert is a sequence modeling module, and each subsequence gets its own representation from its theme expert. All representations are summed with weights to get a final representation. The weights are the routing scores of the most recent item in the full sequence.

### (2) Multi-scale fusion
In (1), each expert focuses on one theme's subsequence and mines deep semantic patterns within that theme. But simply splitting into subsequences may lose global information, so two other kinds of experts are added to complement it:

* Global Expert: processes the full raw sequence and captures global long-term preference.
* Window expert: uses a sliding window to capture short-term local dynamics while staying theme-aligned.
Fusion: the representations from the 3 kinds of networks are summed with learnable weights α, giving a global + theme + short-term multi-view model.

Training strategy: progressive training in 3 stages (backbone warm-up → theme expert warm-up → joint optimization) to keep training stable.
* Stage 1: train the whole backbone network without this paper's module.
* Stage 2: freeze the embedding layer and update only the theme experts and window experts.
* Stage 3: jointly fine-tune all parameters.

### Offline results
Tested on three datasets and four backbones: average AUC +0.68%, GAUC +0.72%, with only about 53% of the compute of a traditional MoE.

## Takeaways:
* The paper finds that computing weights with an MLP does not work well, worse than the codebook. In any scenario with a large number of expert networks where you want each expert to focus more on one aspect, you can try replacing the MLP with a codebook as the weighting network and see whether it helps.
* The codebook entries are centroids of item embedding clusters. This paper clusters sequence items into different clusters, and different expert networks handle different clusters.

## Caveats

## Credibility: offline

## Recommendation: worth a look if you have time

![pic](../imgs/mos.png)
