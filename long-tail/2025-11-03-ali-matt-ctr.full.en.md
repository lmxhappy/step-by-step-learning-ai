# Recommender algorithms (2025.11.3): Alibaba's inference-time feature cold start, CTR +5.3%

> [中文版](2025-11-03-ali-matt-ctr.md)

1 MATT-CTR: Unleashing a Model-Agnostic Test-Time Paradigm for CTR Prediction with Confidence-Guided Inference Paths

Link: https://arxiv.org/pdf/2510.08932

Company: Alibaba

Idea: sampling (approximating a complex problem by random sampling) + ensemble (bagging)

Area: feature cold start + features

Overview: at inference time, the CTR model predicts poorly for very sparse feature combinations. To address this, for each sample the paper selects several high-frequency partial feature combinations, predicts with the existing CTR model, and takes a weighted sum as the final CTR estimate.

(1) Feature combination generation

It borrows the Monte Carlo random-sampling idea to handle inference uncertainty, but makes it more efficient with a confidence mechanism.

Put simply, features are selected one by one recursively (method below). Iterating T steps selects T features that form one feature combination; this combination can be seen as a masked version of the sample's full input features, with the unselected features masked out. This is done K times in parallel, producing K such feature sets.

Selection method: take any unselected feature, combine it with the already-selected features as input, and compute its confidence with a function. Every unselected feature thus gets a confidence; all confidences are normalized, then one feature is chosen via Bernoulli sampling and added to the selected set.

The specific computation is called "Hierarchical Probabilistic Hashing". I didn't read it; interested readers can look into it themselves.

(2) Prediction

For each of the K feature combinations, the full feature vector has the features not in that combination masked out; the resulting vector is fed into the CTR model to compute a CTR. The K CTRs are combined by weighted sum, with weights being the confidences computed by "Hierarchical Probabilistic Hashing".

The ensemble is needed because each feature combination uses only 10-15 features, so accuracy drops; the ensemble recovers it.

A/B: CTR +5.3% on an e-commerce platform.

Takeaways: (1) The idea is very simple and direct: sample several times to get several feature combinations, predict with the existing model, and take a weighted sum. The benefit: no retraining is needed; just some extra work at inference time improves the model. (2) The method is also very flexible: the number of samples and the number of features can be adjusted for different scenarios to strike different balances between effectiveness and efficiency.

Caveats: none

Credibility: production

Recommendation: practically useful
