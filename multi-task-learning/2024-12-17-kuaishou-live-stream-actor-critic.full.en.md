# Recommendation papers (2024.12.17): Kuaishou uses multi-group Actor-Critic reinforcement learning to blend live streams into short-video feeds
> [中文版](2024-12-17-kuaishou-live-stream-actor-critic.md)

> Source: WeChat post https://mp.weixin.qq.com/s/EpDyXfVEF1mlnwo4Q66ZKA, published 2024-12-17.
> Split from the roundup "Recommendation papers (2024.12.17)"; same issue: [Alibaba prioritized MTL](../multi-task-learning/2024-12-17-ali-prioritized-mtl.full.en.md), [Huawei RecSys Arena](../llm4rec/2024-12-17-huawei-recsys-arena.full.en.md), [Kuaishou live-stream Actor-Critic](../multi-task-learning/2024-12-17-kuaishou-live-stream-actor-critic.full.en.md), [TikTok USM survey modeling](../multi-task-learning/2024-12-17-bytedance-usm-survey.full.en.md).

Supervised Learning-enhanced Multi-Group Actor Critic for Live-stream Recommendation

Link: https://arxiv.org/pdf/2412.10381

Company: Kuaishou

Area: mixed ranking + reinforcement learning

Overview: This paper proposes an Actor-Critic reinforcement learning algorithm to solve the live-stream injection problem in mixed short-video and live-stream recommendation. The Actor network learns the live-stream injection policy and outputs the probability of injecting a live stream; the Critic network evaluates how good the current policy is and computes the Q value. It explicitly accounts for user heterogeneity, letting the model learn policies for different user groups.

Credibility: production

Recommendation: read if you have time
