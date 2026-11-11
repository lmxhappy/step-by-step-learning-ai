# 推荐算法(2024.12.18)：支付宝搜索用分布感知鲁棒学习提升LLM相关性建模

> 原文：https://mp.weixin.qq.com/s/-m8X1ILtc_5878LzriS_JQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-18
> 本文拆自《推荐算法paper（2024.12.18）》，同期 2 篇：[支付宝 LLM 相关性建模](../llm4rec/2024-12-18-ant-llm-relevance-robust.md)、[华为类别不平衡图分类](../graph-recommendation/2024-12-18-huawei-cluster-contrastive-graph.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-18-ant-llm-relevance-robust.full.en.md)

Boosting LLM-based Relevance Modeling with Distribution-Aware Robust Learning

网址：https://arxiv.org/pdf/2412.12504

公司：支付宝搜索

方向：搜索+LLM

解读：背景是LLM模型评估搜索相关性，在区分细粒度的相关性（例如强相关、弱相关、无关）时存在困难。方法是从大量的无标签数据里抽取出来分布外（OOD）数据，并将其增强LLM模型。

可信度：生产

推荐等级：值得看看
