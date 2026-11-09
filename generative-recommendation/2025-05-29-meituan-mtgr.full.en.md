# Recommendation algorithms (2025.5.29): Meituan's plan for deploying generative recommendation

> [中文版](2025-05-29-meituan-mtgr.md)

1 MTGR: Industrial-Scale Generative Recommendation Framework in Meituan

Link: https://www.arxiv.org/pdf/2505.18654v2

Company: Meituan

Area: generative recommendation

Overview:

This paper describes deploying HSTU at Meituan. On the model side it makes small innovations; the main innovation is in the inference stage, and that is done in order to ship. The inference stage makes heavy use of Nvidia's strong features to tap and bring out the inference capability of its GPUs.

The small innovations:

(1) Keeping cross features: user features, the history behavior sequence, real-time interactions and candidate features (including cross features) are turned into one unified token sequence, with cross features folded into the candidate tokens.

(2) Group layer normalization: tokens from different domains are normalized by domain group, so that the token distribution within each domain is similar, usually adjusted to mean 0 and variance 1, which aligns the semantic spaces of different domains before self-attention.

(3) Dynamic masking strategy: a way MTGR handles the token sequence, mainly to avoid information leakage while improving model performance. The core idea is to flexibly control which tokens can "see" the information of other tokens, based on token type and temporal relationship.

The innovations in the inference stage:

(1) By integrating the deeply optimized Cutlass-based HSTU kernel provided by Nvidia, variable-length sequence inputs are supported without padding, which greatly improves Attention compute efficiency; single-operator performance is 2-3x that of the Triton version.

(2) Dynamic batch size: each card's batch size is adjusted dynamically by the actual sequence lengths in the data, keeping the compute (total_tokens) roughly the same. Because a few users have very long sequences and most users have fairly short ones, giving each card the same number of users leads to quite different actual compute due to different sequence lengths. And at every step, all cards have to wait for the most heavily loaded card to finish before gradient sync.

(3) TensorRT chosen as the model inference framework: TensorRT is an inference optimization framework from Nvidia, widely used in industry, with strong operator fusion and low-precision quantization capabilities.

The A/B results: conversions +1.22%, click-through rate (CTR) +1.31%. Meanwhile, training cost stays the same and inference cost drops 12%.

Reference: https://static.rainfocus.com/nvidia/gtcs25/sess/1734091150868001dcjp/FinalPresPDF/S74073_1741683377945001YCts.pdf

Takeaways: (1) This is one way to ship generative recommendation: big Chinese companies put most of their effort into the inference stage, and inference optimization means tapping the capabilities of GPUs and their APIs, where DeepSeek's success offers a model to learn from. Leave model innovation to the big foreign companies, and let the big Chinese companies do the following and the deployment well.

Credibility: production

Recommendation: worth a look
