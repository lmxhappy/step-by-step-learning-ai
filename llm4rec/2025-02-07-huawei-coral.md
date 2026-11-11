# 推荐算法(2025.2.7)：华为CoRAL用大模型做协同排序

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484080&idx=1&sn=f2dedc72ef006b9239eb32fc94dafc3d&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-07
> 本文拆自《推荐算法paper（2025.2.7）》第 7 篇。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-07-huawei-coral.full.en.md)

CoRAL: Collaborative Ranking with Large Language Models

网址：https://arxiv.org/pdf/2501.18937

公司：华为

方向：大模型+排序

解读：华为关于大模型排序的最新paper。CoRAL提出了一种基于大语言模型的协同排序方法，通过将大语言模型与协同过滤结合，实现了高效的排序建模。具体来说，CoRAL首先将用户行为序列转换为嵌入向量，然后通过大语言模型，捕捉用户兴趣的变化。同时，CoRAL还引入了一种新的协同排序模块，通过将大语言模型与协同过滤结合，实现了高效的排序建模。

可信度：离线

推荐等级：有时间看看
