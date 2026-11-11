# Recommendation papers (2024.12.17): Huawei RecSys Arena: pairwise evaluation of recommendation results with an LLM judge
> [中文版](2024-12-17-huawei-recsys-arena.md)

> Source: WeChat post https://mp.weixin.qq.com/s/EpDyXfVEF1mlnwo4Q66ZKA, published 2024-12-17.
> Split from the roundup "Recommendation papers (2024.12.17)"; same issue: [Alibaba prioritized MTL](../multi-task-learning/2024-12-17-ali-prioritized-mtl.full.en.md), [Huawei RecSys Arena](../llm4rec/2024-12-17-huawei-recsys-arena.full.en.md), [Kuaishou live-stream Actor-Critic](../multi-task-learning/2024-12-17-kuaishou-live-stream-actor-critic.full.en.md), [TikTok USM survey modeling](../multi-task-learning/2024-12-17-bytedance-usm-survey.full.en.md).

RecSys Arena: Pair-wise Recommender System Evaluation with Large Language Models

Link: https://arxiv.org/pdf/2412.11068

Company: Huawei

Area: LLM ranking

Overview: Inspired by the idea of the LLM Chatbot Arena, this paper proposes the concept of RecSys Arena, where an LLM judge evaluates recommendation results to obtain fine-grained evaluation feedback. Specifically, for each sample, we use the LLM to generate a user profile description from the user's behavior history or readily available user features, and use it to guide the LLM to play the role of that user and judge the relative preference between two recommendation results produced by different models. LLMs are getting smarter and smarter; simulating humans to judge and improve recommendation models is a good approach and direction, meaningful for both research and work :).

Credibility: offline

Recommendation: worth reading
