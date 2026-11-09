# Meta: deep retrieval, business metric +3%

> [中文版](meta-hsnn.md)

### Paper: Hierarchical Structured Neural Network: Efficient Retrieval Scaling for Large Scale Recommendation
### Link: https://arxiv.org/pdf/2408.06653
### Company: Meta
### Idea: clustering
### Area: deep retrieval

## Overview:
The paper proposes MoNN, a deep retrieval model of "three towers + a fusion head". A traditional two-tower retrieval model can only take the dot product of the user embedding and the item embedding at the retrieval stage, so its interaction capability is very limited. The goal is to move the deep user-item crossing capability of the ranking stage efficiently into the retrieval stage, making retrieval both fast and strong.

### Architecture:
* User Tower: handles only user-only features and outputs the user embedding (computed once per query and reused for all subsequent items).
* Item Tower: handles only item-only features, updated asynchronously.
* Interaction Tower (the main highlight): dedicated to ⟨user, item⟩ cross features.
* Fusion head: sits on top of the three towers, fuses their outputs, and produces multiple task-specific logits (click logit, conversion logit, etc.).

The three towers share the Tower Network (MLP + AttentionFM layers), and the fusion head can be a fusion model such as DHEN or DeepFM.

The highlight of this model is the Interaction Tower, and its key is Inverted Index Based Interaction Features. Concretely:
### (1) For item-side features (category, tag, etc.), inverted indexes are built in advance: the key is the feature value, and the value is the item set, plus an embedding aggregated from the item embeddings; the aggregation can be mean or sum. Here the item embeddings come from the item tower.

### (2) The User Tower has already computed the user-side feature, the categories the user recently engaged with = ["sports", "outdoors", "fashion"] (this is a list or an embedding representation)

### (3) Write the user feature directly as a query: the Interaction Tower takes this list as a query and looks it up in the inverted index in parallel:
* Query "sports" → get that category's pre-aggregated dense vector, i.e., the aggregate of the embeddings of all its items from the item tower.
* Query "outdoors" → get the pre-aggregated dense vector c_outdoors
* Query "fashion" → get the pre-aggregated dense vector c_fashion

### (4) Aggregate immediately (Pooling / Aggregation), instantly combining the multiple vectors into one fixed-dimension vector. Pooling can be mean or sum; aggregation can use a lightweight attention network.

Effect: brute-force crossing cost drops from O(user × item) to one query + lightweight aggregation, making complex crossing feasible in real time over an item corpus of hundreds of millions.

### Training:
It includes two kinds of learning:
* Multi-task supervised learning: clicks/conversions are positive labels, and impressions without click/conversion (downsampled) are negatives. These are all labeled data.
* Semi-supervised learning: uses a large amount of unlabeled data (mainly from full impression data or items not shown at the ranking stage), with an unsupervised loss via self-distillation, for debiasing + regularization. There are two workable self-distillation schemes:
    * Snapshot: generate a new snapshot every few minutes; the snapshot is the teacher and the current model is the student; the teacher's soft labels guide the student, with a cross-entropy loss.
    * EMA: the teacher is a model whose weights are a moving average of several past model versions; everything else is the same as the Snapshot scheme.

Labeled data computes both losses; unlabeled data only does semi-supervised learning.

### Online results
MoNN has been deployed as the base model in Meta's ad system for 2+ years; online A/B tests against a traditional two-tower model show a +2.57% business metric gain.

## Takeaways:
* It drops the old "augmentation + contrastive" route outright and uses distillation + unlabeled data instead to get a similar (or even better) denoising/regularization effect.
* It shows peers how Meta uses full impression data to improve its models.
* This is a 2024 paper; the reason for covering it is to lay the groundwork for a write-up of another recent Meta paper.

## Caveats
Some components are named with internal organization names that are not common in the industry, which adds to the reader's burden.

## Credibility: production

## Recommendation: has practical value

![pic](../imgs/hsnn.png)
