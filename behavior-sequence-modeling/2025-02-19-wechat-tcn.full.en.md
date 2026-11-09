# Recommendation algorithm papers, quick read! WeChat's new breakthrough in sequence modeling

> [中文版](2025-02-19-wechat-tcn.md)

1 Introducing Context Information in Lifelong Sequential Modeling using Temporal Convolutional Networks

Link: https://arxiv.org/pdf/2502.12634

Company: WeChat Channels

Area: sequence modeling

Overview: Behavior sequence modeling is an important direction in recommendation. This WeChat work builds on using a TCN to model behavior sequences and brings in context information: for the user's sequence, it computes context-aware embeddings from adjacent items to improve the model.

(1) Background: the key to behavior sequence modeling is the attention module, which derives an interest representation relevant to the candidate item from the sequence. Usually the attention module works point-wise, looking only at the relevance between each single item in the sequence and the candidate item. However, the context information in adjacent items, which would help assess each item's importance more accurately, is not taken into account.

(2) Method: when computing the embedding of a sequence item, take a certain range of items before and after it to form a fixed-length sequence, and pass it through a linear layer to get the corresponding embedding.

Beyond that, tricks such as dynamic networks are also used.

Takeaways: Behavior sequence modeling is a hot direction with lots of papers, and they offer some inspiring ideas. It feels like there is still a lot to be done: multi-behavior, context awareness, long sequences, bringing in other relations, and so on. Keep going, everyone.

Credibility: production

Recommendation: worth a look.
