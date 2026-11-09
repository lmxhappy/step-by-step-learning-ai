# Recommendation papers (2025.7.16): Meta raises relevance of related short-video recommendations by 12%

> [中文版](2025-07-16-meta-related-video-relevance.md)

1 Balancing Semantic Relevance and Engagement in Related Video Recommendations

Link: https://arxiv.org/pdf/2507.09403

Company: Meta

Area: two-tower retrieval + i2i

Overview:

In two-tower retrieval, an auxiliary task is added to increase the content relevance between the candidate item and the trigger item. In fact, the trigger item and the candidate item use a Siamese network (i.e. they share one encoding network).

Background and motivation:

Traditional collaborative filtering (CF) relies on co-engagement signals (such as co-watching), but has limitations: semantic ambiguity — noisy interactions link unrelated videos; popularity bias — popular content dominates and reduces diversity; lack of interpretability — purely behavioral models are opaque.

Method:

Besides the engagement task, a relevance task is added. Both tasks share the same two-tower output embeddings. The auxiliary task just adds a switch on top of the engagement task's loss: only when a pair is both semantically relevant and co-engaged is the switch on, i.e. the loss is active; otherwise the switch is off, i.e. the loss is multiplied by 0, so it becomes 0 loss.

The main-task and auxiliary-task losses are summed with weights and trained jointly to optimize the model.

The key to this method's success — the trigger item tower brings in context, which improves modeling:

In the two towers, one tower is for the trigger item and one for the candidate item. The trigger item tower considers not only the trigger item but also the most recent preceding user behavior sequence (K items the user interacted with). The final embedding concatenates the context sequence with the trigger item's embedding and passes it through a dense layer to get a final embedding (of the same length as the candidate item's).

Besides the auxiliary task, there are two tricks:

(1) Fusing multimodal content features (text and visual embeddings) for richer semantic understanding;

(2) Off-policy correction (OPC) with inverse propensity weighting, which effectively eases popularity bias.

A/B: on Facebook short-video recommendation, semantic relevance rose from 51% to 63%, the share of popular videos fell 13.8%, the top engagement metric rose 0.04%; short-term engagement dropped slightly but long-term retention improved.

Takeaways:

(1) Using multiple signals as labels has been done for many years; adding a loss switch on the main task as an auxiliary task is not rare, but not common either — perfect. Especially in item-to-item tasks, injecting relevance signals into the label is really effective.

(2) Using multiple signals as labels has been done for many years; adding a loss switch on the main task as an auxiliary task is not rare, but not common either — perfect. Especially in item-to-item tasks, injecting relevance signals into the label or the loss is a common recent practice.

Credibility: production

Recommendation: practically useful

<!-- Figure for this article: the page's image element only returned a placeholder; the image link could not be retrieved -->
