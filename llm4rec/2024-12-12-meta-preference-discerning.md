# 推荐算法(2024.12.12)：Meta用LLM从行为序列归纳用户兴趣，增强生成式检索

> 原文：https://mp.weixin.qq.com/s/B33h3N0mVXTEPYl3Snr3oQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-12
> 本文拆自《推荐算法paper（2024.12.12）》，同期 2 篇：[Meta LLM 归纳用户兴趣](../llm4rec/2024-12-12-meta-preference-discerning.md)、[华为 AltFS 特征选择](../feature-selection/2024-12-12-huawei-altfs.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-12-meta-preference-discerning.full.en.md)

Preference Discerning with LLM-Enhanced Generative Retrieval

网址：https://arxiv.org/pdf/2412.08604

公司：meta

方向：行为序列+LLM

解读：通过LLM模型从用户行为序列中归纳出来多个用户兴趣，然后再将其应用到行为序列建模中。关键是从序列中归纳用户兴趣，就是将序列元素的信息，如描述等，和用户的评论信息等拼接起来，送入LLM模型，让其归纳用户兴趣。归纳的用户兴趣也是文字，将文字通过多模态的encoder转成embedding，再做后续的行为序列建模的提升。LLM归纳用户兴趣和意图，是一个很火的做法，最近很多paper都在做，看样子是有效的。这个也好理解，推荐模型比较浅，对用户意图很难建模出来，而我们在使用chatgpt的过程中，很明显的感知到它的智能水平是很高的，对用户意图的理解会更好，在可预见的将来随着大预言模型的发展会更好。

可信度：离线

推荐等级：推荐看看
