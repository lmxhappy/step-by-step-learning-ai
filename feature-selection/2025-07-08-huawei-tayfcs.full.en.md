# Recommendation papers (2025.7.8): Huawei's new feature selection method

> [中文版](2025-07-08-huawei-tayfcs.md)

1 TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems

Link: https://arxiv.org/pdf/2507.03895

Company: Huawei

Area: feature selection

Overview:

A method for feature combination selection. The core goal is to pick important, non-redundant feature combinations out of a huge number of possible ones, and remove unimportant or redundant ones. It consists of:

(1) TayScorer module

It analyzes each feature's contribution to the model prediction with a Taylor expansion, and approximates the importance of higher-order feature combinations via the information matrix equality (IME). There is no need to pre-construct all feature combinations: a single backward pass is enough to estimate feature importance efficiently, cutting time complexity from O(n²) to O(n). The mean of the feature embeddings is chosen as the expansion point (x0) of the Taylor expansion to keep the evaluation fair, and higher-order derivatives are approximated by the outer product of gradients.

(2) LRE (Logistic Regression Elimination) module

It uses logistic regression as a proxy model, adds feature combinations batch by batch with a sliding window, and evaluates their information gain (based on validation-set performance). If a feature combination brings negative gain (i.e. it is redundant), it is removed. This effectively removes redundancy among feature combinations and maximizes the information in the selected ones.

(3) Embedding table optimization

Hashing compresses the embedding tables of feature combinations: feature values beyond a threshold (𝜏 = 5.0e6) are mapped to a fixed-size index, lowering memory cost.

A/B: on a CVR prediction task in an ads platform, conversion rate +13.9% and revenue +0.73%.

Credibility: production

Recommendation: worth a look
