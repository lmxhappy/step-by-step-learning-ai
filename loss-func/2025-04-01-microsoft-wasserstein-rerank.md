# 推荐算法(2025.4.1)：微软用多级相关性数据和Wasserstein损失做重排

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484391&idx=1&sn=ef838380260b349648ae9d9bc69b52ab&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-01（修改于 2025-04-01）
> 本文拆自《推荐算法(2025.4.1)：工业风拉满！Meta、百度、微软的成果》，同期 3 篇：[Meta 位置偏置物品选择](../deep-retrieval/2025-04-01-meta-position-biased-retrieval.md)、[微软 Wasserstein 重排](../loss-func/2025-04-01-microsoft-wasserstein-rerank.md)、[百度 CoRanking 大小模型协作排序](../llm4rec/2025-04-01-baidu-coranking.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-04-01-microsoft-wasserstein-rerank.full.en.md)

Wasserstein Reranking for Multi-level Context Retrieval

网址：https://arxiv.org/pdf/2503.23633

公司：微软

方向：搜索

解读：

针对传统二元标签排序模型的局限性，提出了一种新的方法来提升搜索相关性。传统的二元标签排序方法有以下局限性：

（1）忽略细微相关性差异：除了明确标注为相关的文档外，所有其他文档被同等视为负样本，无法反映实际的相关性程度。

（2）对噪声敏感：二元标签无法充分利用数据的排序信息，且易受标注错误影响。

具体方法：

（1）多级排名上下文生成：

使用 MS MARCO 数据集的查询，提示 LLMs 生成四篇不同相关性级别的合成文档。通过顺序生成确保相关性逐步降低，并使用随机采样（如句子长度和难度）增加多样性。分配标签：{3, 2, 1, 0}，分别对应完美相关到无关。

（2）Wasserstein 距离损失：

不同于 InfoNCE 的逐对比较，Wasserstein 距离考虑整个文档集合的相关性分布。通过比较真实标签分布与预测得分分布，优化模型以更准确地排名文档。

心得：细分数据是一个好方法。模型可以细分（解耦），数据也可以细分。都要往细了做。

可信度：离线

推荐等级：推荐看看
