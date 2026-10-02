Alibaba International pre-trains its ranking model generatively, then transfers it into the ranker. Two-week A/B: GMV +9.85%.

Paper: LazFormer - Scaling Transformers for Industrial Recommendation via Transferable Generative Pre-training (arxiv.org/abs/2609.14978)

Scaling a ranker hits two walls. Sparse params (ID embeddings) and dense params both train from scratch - billions of them, slow and expensive. Pre-train your way out and you hit negative transfer: pre-training features don't line up with ranking-stage ones. LazFormer splits the phases and stitches them with an adapter that does nothing at step zero.

1 - Pre-training

Next-item prediction over chronological interactions: one year, 16M users, 11B tokens. The task isn't the point - what transfers is both parameter sets. Sparse params from the embedding layer and dense params from the blocks both initialize the ranker, then the whole base keeps training.

2 - Making the transfer work

Zero-init residual adapter. Shared features existed in pre-training (item ID, category, shop, brand); ranking-specific ones did not - add-to-cart, order, time gap. Those go through their own GELU FFN, added element-wise, with the up-projection initialized to zero. So at step one the adapter outputs exactly 0, the model is identical to the pre-trained one, and gradients decide the dose.

Serving the length. Newest 1,024 tokens stay intact, older ones sum-pool in groups of 8. Attention is a causal 128-window plus long-range tokens, candidates seeing all history but not each other: 88% sparse, 3.7x faster.

3 - Asymmetric multi-epoch training

Three epochs, handled asymmetrically: each epoch sparse params reset to their pre-trained state while dense params inherit the previous one. Billions of embeddings re-updated on the same data overfit and wash out the representation they arrived with. Not new - it's the one-epoch overfitting folklore CTR teams have lived with for years.

4 - Online

Two weeks against a 3-layer SORT-like ranker: GMV +9.85%, item page views +5.21%, orders +3.38%, buyers +3.70%. A10 QPS drops 12.1%.

Takeaways

- When pre-training and downstream features don't align, you either mutilate features into alignment or skip pre-training. Zero-init adapters are a third path: new features start at "not participating," and training decides the dose. Ports to any two-stage transfer.

Nitpicks

- Figure 1 draws the adapter spanning the whole tokenization layer, so it reads as if candidates pass through too. But add-to-cart, order and time gap are empty for a candidate by construction. Only history items are non-zero.
- GMV +9.85% deserves a discount: the control is a 3-layer model, and the paper never separates "we finally scaled up" from "transfer helped."
- Calling those long-range tokens "global" misleads: global usually means bidirectional, but causality stops earlier history seeing these back. "Full-history queries" is the honest name.

#RecSys #RecommenderSystems #MachineLearning #LLM #AI
