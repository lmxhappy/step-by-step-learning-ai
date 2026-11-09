# Google: up to 1000x faster decoding for generative retrieval, fresh video views up 5%

> [中文版](google-static.md)

### Paper: Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators
### Link: https://arxiv.org/pdf/2602.22647
### Company: Google
### Idea: vectorization
### Area: retrieval, generative recommendation

## Overview:
The paper designs an accelerator-friendly constrained decoding method that turns trie traversal into vectorized sparse matrix operations, achieving O(1) I/O complexity.
Traversing a trie is too slow; this paper speeds it up by 2 to 3 orders of magnitude.

### (1) Labeling nodes
Label the nodes of the generated trie, numbered from 0, where 0 is the root. Note: the nodes in the trie are items.
### (2) Building the full matrix
Place the nodes into a full matrix. Row 0 of the matrix describes the child connections of node 0, row 1 those of node 1, ..., row i those of node i, and so on. In the figure, for example, node B1 is node 4, and row 4 of the matrix describes its connections to its children. It has children C2 and C3, which are node 6 and node 7, so columns 2 and 3 of the matrix hold 6 and 7.
### (3) Building the compressed matrix
As shown in the figure, a mapping "node -- row -- column range -- corresponding values" is built.

### (4) Fast decoding
In the figure, A3 and B1 have already been decoded. When decoding the 3rd position, the probabilities of C1, C2 and C3 are known. Looking up the compressed matrix shows that only C2 and C3 are children, so C1 is masked out, which guarantees that subsequently generated tokens are tokens that exist. Beam search then decides the final tokens.

Other filters can be added during this lookup. The paper builds a 7-day freshness constraint "into" the CSR: the lookup automatically ensures generated SIDs correspond only to fresh items, and old videos are never output.

**A/B**: YouTube Shorts, 7-day fresh video views +5.1%, 3-day fresh video views +2.9%, click-through rate (CTR) +0.15%, strategic user satisfaction +0.15%

## Takeaways:
* Google is firmly bullish on generative recommendation and is working hard to polish every component.

## Caveats
In Figure 1, the final probability of C1 in the upper-right corner should be 0.

## Credibility: production

## Recommendation: practically useful

![alt text](../imgs/VectorizingTrie.png)
