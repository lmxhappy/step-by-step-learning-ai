# 推荐算法(2025.1.3)：快手在频谱域用扩散模型做协同过滤

> 原文：https://mp.weixin.qq.com/s/kx6LdmgPOSOOGQYsyiME_Q
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-03
> 本文拆自《推荐算法paper（2025.1.3）》，同期 2 篇：[快手 S-Diff 频谱域扩散协同过滤](../graph-recommendation/2025-01-03-kuaishou-s-diff.md)、[路透社 NewsHomepages 新闻重要性](../loss-func/2025-01-03-reuters-newshomepages.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-03-kuaishou-s-diff.full.en.md)

S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain

网址：https://arxiv.org/pdf/2501.00384

公司：快手

方向：协同过滤+diffusion

解读：将扩散模型用于在频谱域进行协同过滤。该模型首先将用户交互向量映射到频谱域，并利用图的频率信息对扩散噪声进行参数化，保留低频分量以维持高信噪比。然后，使用条件去噪网络编码用户交互信息，并从噪声数据中恢复真实偏好。

可信度：离线

推荐等级：有时间看看
