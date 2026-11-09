# Recommendation Algorithms (2025.11.20): Alibaba uses diffusion for feature enhancement, CTR +2.1%

> [中文版](2025-11-20-ali-sgctr-diffusion.md)

1 Infer As You Train: A Symmetric Paradigm of Masked Generative for Click-Through Rate Prediction

Link: https://arxiv.org/pdf/2511.14403

Company: Alibaba

Idea: using group information to enhance the individual

Area: feature engineering

Overview:

The paper proposes SGCTR, whose core is a fully symmetric design between the training and inference stages. Put simply: training and inference are considered together, and the training process simulates the prediction flow of the inference stage.

Specifically:

1) A discrete diffusion model is used for feature enhancement, to model correlations between features. For example, by modeling CTR data, it can learn combinational relationships among features: some feature combinations often appear together (e.g. user age and product category), and some rarely appear together (e.g. high-end products and low-income users).

2) In training, the diffusion model learns the statistical distribution of features (group information) through step-by-step noising and denoising.

3) At inference, the group information learned in training is used to correct the input features.

Why do this?

Imagine: an ordinary CTR model usually predicts CTR directly from the input features, which is like judging from local information only. For example, if a user usually buys only low-priced products but today browses a high-end product, the model may predict inaccurately for lack of context. SGCTR, however, uses the group information learned by the diffusion model as a kind of "global view", helping the model better understand this user's behavior pattern.

Iteration process:

1) First, the model's input features are partially masked, e.g. a random portion of features is masked out;

2) Then the model tries to recover the masked features through the diffusion model's denoising process;

3) This is repeated multiple times; the features recovered in each round are the input to the next round, refining step by step;

4) After 𝑇 iterations, the model outputs an optimized feature representation used for CTR prediction.

The benefit is that the model learns during training how to recover useful features from incomplete information, which makes it more robust when facing real data at inference.

A/B: tested on an e-commerce platform from October 16 to 26, 2025; compared with the DGenCTR baseline, SGCTR gives a relative CTR gain of 2.1%.

Takeaways:

(1) A closer look at the paper's idea: the discrete diffusion process essentially has the model learn, during training, the statistical distribution of feature combinations from a large number of samples (the "group information", such as common feature dependencies and plausible combinations), and then at inference use this group-level knowledge to "recover" or "refine" the masked features of an individual sample, improving the CTR prediction accuracy for that single sample. This is indeed a classic way of "using group information to improve the individual", similar to the denoising mechanism in generative models such as diffusion models.

Caveats:

The paper's title makes you think it invents a GR model. It should make clear that this is feature-related work, so it's obvious at a glance.

Credibility: production

Recommendation: practically useful
