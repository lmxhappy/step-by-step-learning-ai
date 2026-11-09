# Kuaishou: two-tower retrieval, ad revenue up to +8%

> [中文版](kuaishou-cs3.md)

### Paper: CS3: Efficient Online Capability Synergy for Two-Tower Recommendation
### Link: https://arxiv.org/pdf/2604.22761v1
### Company: Kuaishou
### Idea: clustering, upstream-downstream consistency, knowledge distillation
### Area: two-tower retrieval

## Overview:
The paper proposes a general, pluggable framework that can be integrated seamlessly into any two-tower architecture (such as DSSM, IntTower, etc.). It achieves "capability synergy" through three modules, covering three areas of work: improvement within a tower, synergy between towers, and reuse of the downstream model, corresponding to three innovations.

### (1) Cyclic Adaptive Structure (CAS): improvement within a tower
Let each tower perceive its own output, to denoise features and revise itself. That is, in the user tower and the item tower, multiple CAS module layers are stacked instead of ordinary dense layers.

How CAS works:
1. Ordinary forward: input x passes through a dense layer to get z.
2. Adaptive reweighting: z passes through another dense layer to get e.
3. Denoising: Hadamard product of input x with 2*e.
4. Cyclic forward: feed it into the dense layer (which shares weights with the dense layer of the ordinary forward), and the output becomes the input of the next layer.

Steps 1 and 2 can be seen as a gate network, similar to EPNet in PEPNet. The difference is that this gate network uses the downstream dense layer with shared weights. EPNet is doing denoising, and so is this paper.

Effect: significantly improves the representation quality and robustness of a single tower, as if each tower got a "self-check mechanism".

### (2) Cross-tower synchronization: synergy between towers
Let the user tower and the item tower perceive each other in real time, to address representation alignment.

For a user, the dynamic cluster center of the embeddings of the positive items they have interacted with serves as an extra input to that user in the user tower. Similarly, for an item, the dynamic cluster center of the embeddings of users who interacted positively with it serves as an extra input to that item in the item tower.

The computation is just an exponential moving average, computed on positive samples only.

In essence it gives the user tower a real-time summary signal of "what the items I liked in the past look like". The item side is similar: what the users who like this item look like.

### (3) Cascade model sharing: reuse across stages
Let the two-tower model borrow knowledge directly from the downstream ranking model, to address inconsistency across stages.

For a specific (user, item) pair, the ranking model computes a strong representation, which is "injected" into the user's persistent representation and the item's persistent representation. Concretely, the two representations (user and item) are each updated by moving average.

From then on, every time the user makes a request, the user tower takes the most recently updated representation as an extra input, as if the user tower "secretly learns" the ranking model's strong preference knowledge about this user's history. The item tower does the same, using the item's representation as an extra input.

Here both positive and negative samples are used. It records the combined signal of "which items the ranking model scored high for this user in the past, and which it scored low", representing the user preference prototype as the ranking model sees it. The tower takes this vector directly as an extra input, which amounts to "secretly learning" the representational power of the stronger ranking model.

### A/B:
Across three major scenarios of Kuaishou's ad system, ad revenue improved by up to 8.36%, while keeping millisecond-level latency.

## Takeaways:
* Cross-tower help and downstream reuse both put all the "knowledge" you can get to use, to improve the retrieval model.
* Cross-tower help and downstream reuse are an effective application of the clustering idea: rather than using the current user-item interaction as cross help, the cluster centers of long-term accumulated interaction information help each other.
* Downstream reuse forcibly takes the knowledge of the complex downstream ranking model and uses it for retrieval.

## Credibility: production

## Recommendation: has practical value

![pic](../imgs/cs3.png)
