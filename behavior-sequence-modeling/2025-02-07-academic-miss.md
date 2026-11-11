# 推荐算法(2025.2.7)：何向南团队MISS多兴趣自监督序列推荐

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484080&idx=1&sn=f2dedc72ef006b9239eb32fc94dafc3d&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-07
> 本文拆自《推荐算法paper（2025.2.7）》第 4 篇。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-07-academic-miss.full.en.md)

MISS: Multi-Interest Self-Supervised Learning for Sequential Recommendation

网址：https://arxiv.org/pdf/2501.19207

公司：何向南。虽然带着meta的作者，但是不是一作和通讯作者。相比，何向南的名声更大。

方向：序列推荐

解读：这篇paper提出了一种多兴趣自监督学习方法，通过将用户行为序列转换为多个兴趣表示，然后通过自监督学习，捕捉用户兴趣的变化。具体来说，MISS首先将用户行为序列转换为嵌入向量，然后通过多兴趣提取模块，提取用户的多个兴趣表示。同时，MISS还引入了一种自监督学习模块，通过将用户行为序列与兴趣表示结合，实现了高效的序列建模。

可信度：离线

推荐等级：有时间看看
