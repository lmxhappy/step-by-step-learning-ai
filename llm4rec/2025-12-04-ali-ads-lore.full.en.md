# Recommendation Algorithms (2025.12.4): Alibaba Ads, three years in the making, search relevance metric +27%

> [中文版](2025-12-04-ali-ads-lore.md)

1 LORE: Lessons Learned from Three Years of Large Language Model Development for E-commerce Search Relevance

Link: https://arxiv.org/pdf/2512.03025

Company: Alibaba Ads

Area: LLM + search

Overview:

Search relevance is a very important foundational area. An interesting fact: over these 3 years, Alibaba Ads produced 3 pieces of work, one after another, corresponding to the 3 stages of LLMs in search relevance: discriminative, generative, and reinforcement learning. They are LREF, LREF-M and LORE. This paper is a summary of those 3 years of work, so it does not propose a new method; instead it summarizes the experience gained doing that work.

Background:

Ads, search and recommendation all use semantic models for relevance. After generative techniques appeared, all of these areas saw new developments.

The application of LLMs to search relevance has gone through 3 main stages:

(1) Discriminative

This paper models query-item relevance with a multi-view model, at several semantic granularities. It corresponds to the first stage of LLMs in search relevance: discriminative models.

(2) Generative

This paper turns relevance prediction into a generative task. Concretely, the model takes the query and the item's title, category, shop, price and so on, and outputs a relevance score (between 0 and 1). It corresponds to the second stage of LLMs in search relevance: generative models.

(3) Reinforcement learning

This paper optimizes the relevance model with reinforcement learning. Concretely, an initial model is first trained with supervised fine-tuning, then further optimized with reinforcement learning. The reward in the RL stage is determined jointly by the relevance score and other business metrics. It corresponds to the third stage of LLMs in search relevance: reinforcement learning models.

Insights:

(1) The reasoning ability of LLMs is very useful for search relevance. Using LLM chain-of-thought (CoT) for relevance judgments not only works well but also makes the judgment process explainable. This provides a qualitative analysis framework for future domain-specific LLMs.

(2) Multimodal data helps LLM search relevance. The authors tried using a multimodal large model (VLM) to improve relevance judgment. But experiments showed the VLM was no better than a text-only model, because VLM training focuses on visual fusion and may sacrifice depth in pure-text reasoning.

Lessons from the SFT stage (supervised fine-tuning):

1. Supervised fine-tuning with chain-of-thought works better than fine-tuning to output the answer directly. Training and inference cost more, but the gain is clear. The authors found that using CoT for search relevance judgments not only works well but also makes the judgment process explainable.

2. Feature engineering still matters. Even with an LLM, the choice of input features has a big effect on results. A single feature (e.g. only the product title) may miss information and make the model unstable on borderline cases; adding more relevant features gives richer context and makes the model more robust.

3. Data quality matters more than data quantity. The authors found that cleaned, high-quality data works better than a large amount of low-quality data. Especially for a task like search relevance that needs fine-grained judgment, data quality has an even bigger effect.

4. Evaluation methods need to keep up. Traditional evaluation methods (e.g. accuracy, AUC) may not fully reflect how well an LLM performs. The authors suggest combining human evaluation with automated evaluation to assess the model comprehensively.

Lessons from the RL stage (reinforcement learning):

1. Reward design is key. The reward function has to consider both the relevance score and business metrics. Optimizing only the relevance score may hurt business metrics; optimizing only business metrics may make relevance judgments inaccurate. A balance between the two is needed.

2. The effect of reinforcement learning varies a lot across tasks. On some tasks (such as math or code generation), RL works very well; but on a task like search relevance that requires subjective judgment, its effect is relatively limited. The authors think this may be because the reward signal for search relevance is fairly sparse and hard to optimize effectively with RL.

3. Training stability is a challenge. Model performance may fluctuate during RL training. The authors suggest a smaller learning rate and more training steps to improve stability.

4. Human feedback still matters. Even with RL, human feedback remains an important way to improve the model. The authors suggest combining human feedback with automated rewards to optimize the model comprehensively.

Takeaways:

(1) Work that shares experience and insights is of immense merit.

Credibility: production

Recommendation: worth a look
