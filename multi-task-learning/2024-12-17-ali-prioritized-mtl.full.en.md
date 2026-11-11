# Recommendation papers (2024.12.17): Alibaba Search: tuning-free prioritized multi-task learning with Lagrangian differential multiplier methods
> [中文版](2024-12-17-ali-prioritized-mtl.md)

> Source: WeChat post https://mp.weixin.qq.com/s/EpDyXfVEF1mlnwo4Q66ZKA, published 2024-12-17.
> Split from the roundup "Recommendation papers (2024.12.17)"; same issue: [Alibaba prioritized MTL](../multi-task-learning/2024-12-17-ali-prioritized-mtl.full.en.md), [Huawei RecSys Arena](../llm4rec/2024-12-17-huawei-recsys-arena.full.en.md), [Kuaishou live-stream Actor-Critic](../multi-task-learning/2024-12-17-kuaishou-live-stream-actor-critic.full.en.md), [TikTok USM survey modeling](../multi-task-learning/2024-12-17-bytedance-usm-survey.full.en.md).

No More Tuning: Prioritized Multi-Task Learning with Lagrangian Differential Multiplier Methods

Link: https://arxiv.org/pdf/2412.12092

Company: Alibaba Search

Area: Multi-Task Learning

Overview: This paper proposes a novel multi-task learning framework that uses Lagrangian Differential Multiplier Methods for step-by-step multi-task optimization. The method aims to improve the performance of high-priority tasks while avoiding interference from other tasks. Its main advantage is that it optimizes multiple objectives automatically, without hyperparameters to balance the different tasks, which removes the need for manual tuning. The step-by-step optimization works as follows: (1) Optimize the high-priority task: first optimize the highest-priority task alone until it reaches its best performance. (2) Optimize the lower-priority tasks: next, optimize the other, lower-priority tasks one by one; during this optimization, the Lagrange multiplier method is used to maintain the performance of the high-priority task, i.e., a penalty term is added to the objective function to ensure that optimizing the lower-priority tasks does not hurt the high-priority task.

Credibility: production

Recommendation: worth reading
