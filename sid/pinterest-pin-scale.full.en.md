# Pinterest: wiring Semantic IDs into two-tower retrieval, fresh repins +3.67%

> [中文版](pinterest-pin-scale.md)

### Paper: Pin-SCALE: Semantic Cascading and Alignment Learning for Engagement-Aware IDs in Cold-Start Recommendations
### Link: https://doi.org/10.1145/3805712.3808507
### Company: Pinterest (SIGIR '26)
### Idea: collaborative signal running through SID generation, aggregation and training
### Area: Semantic IDs, two-tower retrieval

## Overview

Everyone agrees Semantic IDs solve cold start. This paper wires item SIDs into a two-tower retrieval model and tries to lift cold-start recommendation quality with them. Fully deployed across Closeup, Home Feed and Search, serving billions of requests a day.

The method splits into three steps along the pipeline: how the SID is generated, how it enters the tower, and how the tower is trained.

![pic](../imgs/pinterest-pin-scale/framework.png)

On the left, Pinterest's two entry points (home feed and search) leading into Related Pins. The three boxes on the right map onto the three sections below.

> "Repin" is Pinterest's core engagement action — saving someone else's image to your own board. "Fresh" throughout means newly published content with no interaction history.

## 1 Generating the SID: put collaborative signal into the quantizer's input

Standard practice derives SIDs from content alone. Pin-SCALE **swaps the representation fed to the quantizer** for one that already encodes collaborative signal — the item embedding from OmniSAGE, Pinterest's 2025 GNN that aggregates neighbours over the user-item interaction graph.

Against an in-house text LLM embedding and against PinCLIP (image-text aligned multimodal), **OmniSAGE wins decisively**: fresh impressions +1.72%. The paper also offers a metric you can check offline: **the higher the first-level codebook entropy, the better the online result**. OmniSAGE sits at 10.93 against a ceiling of 11, with 100% codebook utilisation; PinCLIP fusion is at 7.42 with only 9.7% utilisation. You can screen out weak candidates without spending an experiment slot.

Note: injecting collaborative signal causes **SID drift**, so the quantizer needs retraining more often.

## 2 Feeding the tower: hierarchical cascading pooling

A SID is a sequence of tokens; the tower wants a single item vector. Pin-SCALE uses **a fusion network that merges coarse to fine**: $h_\ell = \mathrm{MLP}_\ell([h_{\ell-1}; e_{\ell+1}])$ with $h_0 = [e_{raw}; e_1]$ — start from the raw content embedding and the coarsest token, then fold in finer tokens layer by layer.

Where the SID actually sits differs across the four surfaces:

* **Closeup** (detail-page i2i): both sides are items, so both towers carry SIDs
* **Search** (query-to-item): item side only
* **Home Feed** (user-to-item): item side only
* **Shopping retrieval** (user + query to item): item side only

## 3 Training the tower: multi-view contrastive learning

Two-tower models commonly use contrastive learning to sharpen representations, and this paper is no exception. Two views:

**Cross-tower alignment (Q2C).** Two-tower models suffer from representation divergence — semantically close things land in two unrelated spaces. The fix is to treat **the same item passed through the query tower and through the candidate tower as a positive pair**. The elegant part: a new pin with zero engagement still gets aligned, as long as it appears as a candidate in one batch and as a query in another — **it needs no interaction history at all**.

Note: Q2C only works on homogeneous i2i surfaces like Closeup. On the heterogeneous surfaces (Search, Home Feed, shopping) it is not used.

**Co-occurrence alignment (Q2Q / C2C).** Two queries that share a candidate within a batch form a positive pair. A cold item has no engagement of its own, but its co-occurrence with popular items is supervision you already have.

The four surfaces train independent models. Within each, any two candidates sharing a left-side query (a search term or an item) form a positive pair, and symmetrically any two queries sharing a candidate do too.

## What each design repairs

* **Engagement-aware tokenization fills an information gap.** A content-only SID conflates content similarity with reaction similarity. Under cold start this bites hardest: a new item can get an ID that is semantically sensible yet says nothing about engagement.
* **Cascading pooling fills a structural gap.** RQ-VAE residuals are ordered — level 1 explains the most variance, later levels refine. Prior work treats tokens as an unordered set (concatenation, mean pooling, n-gram hashing, SentencePiece) and throws the hierarchy away. Attention pooling only partly recovers it: processing tokens independently, it may forge a strong link between first and last while missing the dependency chain that defines a SID.
* **Contrastive learning fills an objective gap.** SID embeddings fed into a deep network hit vanishing gradients, and long-tail SIDs have it worse: they appear rarely, update rarely, and their representations stay noisy. Contrastive losses supply dense label-free signal exactly where supervision runs out.

## A/B results

On Closeup, cross-tower alignment lifts sitewide fresh repins by 3.67%. Home Feed retrieval gains DAU +0.05%. Search and shopping see gains in fresh long clicks and repins respectively.

## Takeaways
* **"Codebook utilisation and entropy predict online performance" is the cheapest finding in the paper.** Screening candidates without shipping them is exactly the kind of offline proxy the industry undervalues — most teams still pick on intuition and go straight online.
* There are two ways to push collaborative signal into SIDs: **change the representation you feed the quantizer** (this paper), or **let the recommendation loss backprop into the codebook** (end-to-end joint training). The first is far lighter and does not break the codebook's frozen contract; the price is more frequent retraining.
* **Eight hierarchy levels is unusually deep.** Generative retrieval of the TIGER lineage mostly uses three or four. Eight shows up in large-scale discriminative retrieval occasionally, but it is rare.

## Caveats
* **The ablations all live on one surface.** Only cascading pooling was reported cross-surface; engagement-aware tokenization and contrastive learning have Closeup data only. The abstract says the *framework* was validated on multiple surfaces, which reads as broader than the evidence given.
* **Q2C depends on query and candidate being the same type** — and it is the single largest contributor (+3.67%). Closeup is pin-to-pin so it works; on user-to-item surfaces the loss cannot even be constructed. Two of the three methods generalise, not three.
* **SID drift is named but never quantified.** How much more often does the quantizer need retraining? At what cost? How far can it drift before online metrics suffer? Those are the numbers anyone copying this needs, and none are given.
* **Five pages leaves a lot unsaid**: the MLP structure inside cascading pooling, and separate online numbers for Q2Q versus C2C (Table 4 reports only the merged "co-occurrence alignment").

## Credibility: production — fully deployed across Pinterest surfaces, billions of requests daily, the three method comparisons all run as online A/B

## Recommendation: worth adopting
