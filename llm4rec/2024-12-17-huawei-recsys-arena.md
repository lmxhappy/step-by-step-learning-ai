# 推荐算法(2024.12.17)：华为RecSys Arena：用LLM裁判成对评估推荐结果

> 原文：https://mp.weixin.qq.com/s/EpDyXfVEF1mlnwo4Q66ZKA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-17
> 本文拆自《推荐算法paper（2024.12.17）》，同期 4 篇：[阿里优先级多任务学习](../multi-task-learning/2024-12-17-ali-prioritized-mtl.md)、[华为 RecSys Arena](../llm4rec/2024-12-17-huawei-recsys-arena.md)、[快手直播混排 Actor-Critic](../multi-task-learning/2024-12-17-kuaishou-live-stream-actor-critic.md)、[TikTok USM 问卷建模](../multi-task-learning/2024-12-17-bytedance-usm-survey.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-17-huawei-recsys-arena.full.en.md)

RecSys Arena: Pair-wise Recommender System Evaluation with Large Language Models

网址：https://arxiv.org/pdf/2412.11068

公司：华为

方向：LLM排序

解读：受到LLM聊天机器人竞技场（Chatbot Arena）理念的启发，本文提出了RecSys Arena的概念，LLM裁判对推荐结果进行评估，以获得细粒度的评估反馈。具体而言，对于每个样本，我们使用LLM根据用户行为历史或现成的用户特征生成用户画像描述，并以此为依据引导LLM扮演该用户的角色，评估由不同模型生成的两个推荐结果的相对偏好。LLM越来越智能，模拟人来判断、提升推荐模型，是一个不错的做法和方向，不管是科研和工作都有意义：）。

可信度：离线

推荐等级：推荐看看
