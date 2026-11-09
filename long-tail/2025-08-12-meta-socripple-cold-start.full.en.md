# Recommender algorithms (2025.8.12): Meta's item cold start, really really simple

> [中文版](2025-08-12-meta-socripple-cold-start.md)

1 SocRipple: A Two-Stage Framework for Cold-Start VideoRecommendations

Link: https://arxiv.org/pdf/2508.07241

Company: Meta

Area: item cold start

Overview:

Item cold start is handled with two methods. First, trial distribution of new items based on social relations; second, u2u2i, where the retrieved items are ordered by rules and the top ones are taken as one retrieval channel's results.

(1) Trial distribution based on social relations

That is, distribute the item to the followers of its creator. This yields initial interaction data.

(2) u2u2i retrieval for new items

1) u2u: use a pre-trained two-tower model to get the current user's embedding, and search the top similar users from the ANN index.

2) u2u2i:

The new items these similar users interacted with in the last 24 hours form a set. Ordering: remove already-shown videos, apply rules/heuristics, and sort by a weighted score (support count, similarity, freshness), with weights tuned online via A/B tests.

Relation between the two stages: without the first stage, the second stage finds fewer new items.

A/B: two weeks, retention +0.22%; the distribution share of items ≤24 hours old +36%.

Takeaways:

(1) Some companies have had the first stage for a long time but never published it as a paper. Meta made it public and showed that the trick works.

(2) Most companies try u2u2i too. Meta getting results with it is also an encouragement.

(3) A retrieval channel like this most likely needs a boost in rerank to surface, because the ranking model may be biased against these items.

Credibility: production

Recommendation: practically useful
