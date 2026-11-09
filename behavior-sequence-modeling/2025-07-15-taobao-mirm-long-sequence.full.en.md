# Recommendation Paper Pick (2025.7.15): Taobao's new long-sequence retrieval method grows GMV by 4 points

> [中文版](2025-07-15-taobao-mirm-long-sequence.md)

### Paper 1: MIRM: Towards the Modeling of Interest Retrieval and Matching for Long-term User Behavior Sequence
### Link: https://arxiv.org/pdf/2507.10714
### Company: Taobao
### Area: long-sequence retrieval

## Overview:

Background: traditional recommender systems rely on recent behavior, or handle long behavior sequences by simply extending sequence length, but they are limited by compute cost and memory. Long-sequence retrieval is a direction for solving the long-sequence problem in industry: user behavior sequences are often tens of thousands long, and the question is how to extract interests efficiently and retrieve relevant items accurately.

Core idea: the MIRM framework aims to solve the efficiency and accuracy problems of long-sequence retrieval. It decomposes the long sequence into multiple subsequences and uses several retrieval strategies to extract the user's long-term interests, improving the retrieval quality of the recommender system. Specifically:

(1) Interest modeling and matching with the MIRM model. A long-sequence retrieval system usually has two components: interest modeling and interest matching. MIRM first decomposes the user's long behavior sequence into multiple subsequences based on behavior category and time interval. Then, through retrieval strategies (including nearest neighbor and maximum inner product search), it extracts the target user's interest information and matches it with candidate items.

(1) The pooled result of the recent behavior sequence is used as the query; from the subsequences above, the ones with the same category as the target item are selected for Target Attention, producing the corresponding embedding, which serves as the long-sequence interest embedding. Note that at inference, multiple interest embeddings are produced with the top-k subsequences. At the same time, the recent behavior sequence goes through self-attention to get one short-term interest embedding.

The two kinds of interest vectors, together with the target item and so on, are used to compute the match probability between the user and the target item, and the whole model is optimized through a weighted sum of two losses. Note that the target items in the samples are all positives the user interacted with; some negatives are sampled from the target item's category and are used only in the denominator when computing match probability.

(2) A Pointer-Generator Interest Network (PGIN) predicts the top-K interest categories. Based on the top-K categories, parallel ANN search is run: the long-term interest embeddings retrieve only from their corresponding categories, while the short-term interest embedding retrieves from the full candidate pool. This ensures offline consistency and reduces compute overhead.

A/B results: run for three weeks on Taobao's flash-sale mini-program, clicks +5.54%, orders +11.01%, GMV +4.03%. Overall system response time increased by about 15ms.

## Takeaways:

(1) Long-sequence retrieval is an important direction for handling users' long behavior sequences in recommender systems. The MIRM framework decomposes the long sequence into multiple subsequences and combines several retrieval strategies, achieving efficient interest modeling and matching. (2) The long-term interest extraction method in this paper worked well on a platform like Taobao, and the PGIN design also provides an effective solution for parallel ANN search.

## Credibility: production

## Recommendation: worth a look

<!-- Figure for this article: the page's image element returned no image link, so it could not be archived -->
