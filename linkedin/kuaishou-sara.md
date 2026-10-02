Kuaishou turns what users say about why they like a streamer into ranking features. Hate feedback down 8.16%.

Paper: SARA - Scaling Articulated Rationales for MLLM-based Recommendation, Kuaishou (arxiv.org/abs/2609.17639)

Clicks, watch time and dislikes record what users did. What users articulate - nostalgia for a hometown, a sense of company, disgust at hard-selling - records why. The catch is volume: that signal is sparse, uneven, and covers few authors.

1 - Collect

A survey asks live-stream users to pick a polarity and explain in free text, incentives tied to quality. An offline Agent Judge filters on six dimensions: coherence, relevance, specificity, safety, polarity consistency, grounding. Pass rate is 9.8%: 1.92M raw rationales become 187,532 across 86,564 authors, aggregated into per-author semantic labels.

2 - Scale

SARA-7B (Qwen2.5-VL-7B) takes sampled frames, stream metadata, speech transcripts and viewer comments, plus an instruction fixing polarity. SFT on the curated set, then QR-DPO: sample candidates, have the Agent Judge rank them, pair best against worst, optimize. That lifts specificity and grounding - rationales tie concrete activities to viewing interest, not "the host is fun."

Coverage goes from 86K authors to 10M. Crucially the rationales are author-level, not user-author-level, so they precompute offline, refresh daily, and cost nothing at serving.

3 - Fuse

Positive rationales are an alignment target, not an input feature. User profile, author description and rationale share one encoder; the user-author interaction representation is tied to the rationale embedding by a mutual-information-maximization loss, pulling "why this pair matches" toward the stated reason. It reaches the tower through a low-rank adaptor into the MMoE, plus a gated residual so the model sets the dose.

Negative rationales become hierarchical semantic IDs where semantically close complaints share prefixes. Target-aware attention uses the candidate author as query and the user's previously-disliked authors as keys/values. A dislike of author A transfers to an unseen author B in the same complaint cluster, before B accumulates any negatives.

Online

Two independent A/B tests, 1% traffic each, against a ranker that already had multimodal features. Positive path: watch time +0.99%, clicks +0.34%, follows +0.62%. Negative path: dislikes -8.16%, reports -0.44%. Deployed 30+ days within the existing latency budget.

Takeaways

- The moat is the 9.8% pipeline, not the model. Most teams stall for lack of a seed set, not inability to tune an MLLM.
- Author-level rather than pair-level rationales is what makes this shippable: 10M authors times users would never precompute.
- The negative path far outperforms the positive one: complaints cluster into a few types (noise, hard-selling, unsafe), so semantic transfer works. Positive preference is more idiosyncratic.

#RecSys #RecommenderSystems #MachineLearning #LLM #AI
