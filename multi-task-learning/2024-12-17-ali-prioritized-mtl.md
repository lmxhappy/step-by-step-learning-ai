# 推荐算法(2024.12.17)：阿里搜索用拉格朗日微分乘子法做免调参的优先级多任务学习

> 原文：https://mp.weixin.qq.com/s/EpDyXfVEF1mlnwo4Q66ZKA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-17
> 本文拆自《推荐算法paper（2024.12.17）》，同期 4 篇：[阿里优先级多任务学习](../multi-task-learning/2024-12-17-ali-prioritized-mtl.md)、[华为 RecSys Arena](../llm4rec/2024-12-17-huawei-recsys-arena.md)、[快手直播混排 Actor-Critic](../multi-task-learning/2024-12-17-kuaishou-live-stream-actor-critic.md)、[TikTok USM 问卷建模](../multi-task-learning/2024-12-17-bytedance-usm-survey.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-17-ali-prioritized-mtl.full.en.md)

No More Tuning: Prioritized Multi-Task Learning with Lagrangian Differential Multiplier Methods

网址：https://arxiv.org/pdf/2412.12092

公司：阿里搜索

方向：Multi-Task Learning

解读：本文提出了一种新颖的多任务学习框架，采用拉格朗日微分乘子法（Lagrangian Differential Multiplier Methods）进行逐步的多任务优化。该方法旨在提升高优先级任务的性能，同时避免其他任务的干扰。其主要优势在于能够自动优化多个目标，而无需为不同任务平衡超参数，从而消除了手动调优的需求。通过逐步优化的方法:（1）优化高优先级任务: 首先单独优化最高优先级任务，直到达到最优性能。（2）优化低优先级任务:接下来，依次优化其他优先级较低的任务，在优化过程中，使用拉格朗日乘数法来维持高优先级任务的性能，即目标函数中增加了惩罚项，保证低优先级任务的优化不会影响到高优先级任务。

可信度：生产

推荐等级：推荐看看
