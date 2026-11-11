# 推荐算法(2024.11.25)：华为MIRRN多粒度检索与精炼长期用户兴趣

> 原文：https://mp.weixin.qq.com/s/Tui8ZesqJjpCHRf2hUCRig
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-25
> 本文拆自《推荐算法paper（2024.11.25）》，同期 2 篇：[华为 MIRRN 多粒度兴趣检索与精炼](../behavior-sequence-modeling/2024-11-25-huawei-mirrn-multi-granularity-interest.md)、[华为 LIBER 大模型终身行为建模](../behavior-sequence-modeling/2024-11-25-huawei-liber-llm-lifelong.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-25-huawei-mirrn-multi-granularity-interest.full.en.md)

Multi-granularity Interest Retrieval and Refinement Network for Long-Term User Behavior Modeling in CTR Prediction

网址：https://arxiv.org/pdf/2411.15005

公司：华为

方向：行为序列建模

解读：使用了3种建模方法（如sim方法、gru/sumpool方法、聚类），获取用户在不同粒度上的兴趣子序列。用一种新颖的多头傅里叶变换器处理3个子序列，获得了target-aware,local-aware, global-aware等3种interest。最后，用多头目标注意力机制融合这3种表征，获得了用户的兴趣表征。特别的，通过一种叫密度峰值聚类算法（DPC）聚类方法，能够处理几千长的序列。

可信度：生产

推荐等级：值得看看
