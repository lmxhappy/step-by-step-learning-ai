# 推荐算法(2024.12.18)：华为用聚类引导的对比学习做类别不平衡图分类

> 原文：https://mp.weixin.qq.com/s/-m8X1ILtc_5878LzriS_JQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-18
> 本文拆自《推荐算法paper（2024.12.18）》，同期 2 篇：[支付宝 LLM 相关性建模](../llm4rec/2024-12-18-ant-llm-relevance-robust.md)、[华为类别不平衡图分类](../graph-recommendation/2024-12-18-huawei-cluster-contrastive-graph.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-18-huawei-cluster-contrastive-graph.full.en.md)

Cluster-guided Contrastive Class-imbalanced Graph Classification

网址：https://arxiv.org/pdf/2412.12984

公司：华为

方向：GNN。本文虽然与推荐算法无关，但是GNN可以用用于推荐。

解读：该方法将聚类的思想融入对比学习中，以增强类别不平衡图分类的效果。（1）将每个多数类别的图聚类成多个子类，确保它们与少数类别具有相似的大小，从而缓解类别不平衡的问题，这样可以充分探索多数类别内的语义子结构。 （2）利用 Mixup 技术来合成新样本并丰富每个子类的语义信息，并利用监督对比学习来分层学习有效的图表示，这样有效地缓解对少数类别的过度关注。

可信度：离线

推荐等级：值得看看
