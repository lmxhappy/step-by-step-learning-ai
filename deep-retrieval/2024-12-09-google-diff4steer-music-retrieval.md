# 推荐算法(2024.12.9)：谷歌用扩散模型做生成式音乐检索

> 原文：https://mp.weixin.qq.com/s/eZQ7rTJH49dpwf84arAnAQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-09
> 本文拆自《推荐算法paper（2024.12.9）》，同期 3 篇：[Meta Epinet 内容冷启动](../long-tail/2024-12-09-meta-epinet-cold-start.md)、[沃尔玛语义检索](../deep-retrieval/2024-12-09-walmart-semantic-retrieval.md)、[谷歌 Diff4Steer 音乐检索](../deep-retrieval/2024-12-09-google-diff4steer-music-retrieval.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-09-google-diff4steer-music-retrieval.full.en.md)

Diff4Steer: Steerable Diffusion Prior for Generative Music Retrieval with Semantic Guidance

网址：https://arxiv.org/pdf/2412.04746

公司：谷歌

方向：召回+diffusion

解读：基于扩散的生成方法（diffusion）在音乐检索中的应用。生成的嵌入质量较高，能够反映数据的潜在分布，同时能够融入用户偏好建模中的不确定性，从而提高检索效果，既能保证多样性，又具有竞争力的检索和排序性能。

可信度：离线

推荐等级：推荐看看
