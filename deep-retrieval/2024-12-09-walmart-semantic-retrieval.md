# 推荐算法(2024.12.9)：沃尔玛混合语义检索系统

> 原文：https://mp.weixin.qq.com/s/eZQ7rTJH49dpwf84arAnAQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-09
> 本文拆自《推荐算法paper（2024.12.9）》，同期 3 篇：[Meta Epinet 内容冷启动](../long-tail/2024-12-09-meta-epinet-cold-start.md)、[沃尔玛语义检索](../deep-retrieval/2024-12-09-walmart-semantic-retrieval.md)、[谷歌 Diff4Steer 音乐检索](../deep-retrieval/2024-12-09-google-diff4steer-music-retrieval.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-09-walmart-semantic-retrieval.full.en.md)

Semantic Retrieval at Walmart

网址：https://arxiv.org/pdf/2412.04637

公司：沃尔玛

方向：向量召回

解读：提出了一种混合检索系统，结合了传统的倒排索引和神经网络检索，它引入了新方法来选择训练中的负样本，提升了对复杂尾部查询的处理能力。另外，还分享了在电商环境中成功部署该系统的经验教训。

可信度：生产

推荐等级：推荐看看
