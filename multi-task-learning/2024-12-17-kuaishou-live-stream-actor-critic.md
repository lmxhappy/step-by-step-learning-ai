# 推荐算法(2024.12.17)：快手用多组Actor-Critic强化学习做短视频与直播混排

> 原文：https://mp.weixin.qq.com/s/EpDyXfVEF1mlnwo4Q66ZKA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-17
> 本文拆自《推荐算法paper（2024.12.17）》，同期 4 篇：[阿里优先级多任务学习](../multi-task-learning/2024-12-17-ali-prioritized-mtl.md)、[华为 RecSys Arena](../llm4rec/2024-12-17-huawei-recsys-arena.md)、[快手直播混排 Actor-Critic](../multi-task-learning/2024-12-17-kuaishou-live-stream-actor-critic.md)、[TikTok USM 问卷建模](../multi-task-learning/2024-12-17-bytedance-usm-survey.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-17-kuaishou-live-stream-actor-critic.full.en.md)

Supervised Learning-enhanced Multi-Group Actor Critic for Live-stream Recommendation

网址：https://arxiv.org/pdf/2412.10381

公司：快手

方向：混排+强化学习

解读：本文提出一种Actor-Critic强化学习算法，用于解决短视频和直播混合推荐场景中的直播流注入问题。Actor网络学习直播注入的策略，输出是否注入直播流的概率；Critic 网络评估当前策略的好坏，计算 Q 值。显式地考虑了用户异质性，让模型学习针对不同用户组的策略。

可信度：生产

推荐等级：有空看看
