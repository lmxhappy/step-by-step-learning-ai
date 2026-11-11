# 推荐算法(2024.12.12)：华为AltFS：用LLM做推荐模型的特征选择

> 原文：https://mp.weixin.qq.com/s/B33h3N0mVXTEPYl3Snr3oQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-12
> 本文拆自《推荐算法paper（2024.12.12）》，同期 2 篇：[Meta LLM 归纳用户兴趣](../llm4rec/2024-12-12-meta-preference-discerning.md)、[华为 AltFS 特征选择](../feature-selection/2024-12-12-huawei-altfs.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-12-huawei-altfs.full.en.md)

AltFS: Agency-light Feature Selection with Large Language Models in Deep Recommender Systems

网址：https://arxiv.org/pdf/2412.08516

公司：华为

方向：特征选择+LLM

解读：通过LLM模型选出特征集里重要的特征。paper以3个公知的public数据集，让GPT4等告知特征重要性，效果不错。GPT4等LLM模型对这些数据集很熟悉了。但是对于私有生产的数据不熟悉的大模型，估计就没那么好使了，那么就要想办法灌入私有数据，训练LLM，再大模型替我们选特征了。

可信度：离线

推荐等级：有时间看看
