# 推荐算法(2025.5.8)：eBay用层次图神经网络+LLM做广告关键词推荐

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484636&idx=1&sn=8ab523471efbb4cd3e03edb68ca79dea&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-05-08
> 本文拆自《推荐算法(2025.5.8)：ebay、字节和小红书的精彩工作》，同期 3 篇：[eBay 层次 GNN+LLM 广告关键词推荐](../deep-retrieval/2025-05-08-ebay-ad-keyword-gnn-llm.md)、[字节 S3 自监督长序列建模](../behavior-sequence-modeling/2025-05-08-bytedance-s3-long-sequence.md)、[小红书动作导向序列建模](../behavior-sequence-modeling/2025-05-08-xiaohongshu-action-oriented-seq.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-05-08-ebay-ad-keyword-gnn-llm.full.en.md)

Enhancing Ad Keyword Recommendation via Hierarchical Graph Neural Network and LLM

网址：https://arxiv.org/pdf/2504.21187

公司：eBay

方向：关键词召回

解读：本文提出了一种基于图神经网络（GNN）和大语言模型（LLM）的广告关键词推荐方法，用于解决卖家在创建广告时难以选择有效关键词的问题。通过构建卖家-广告-关键词的异构图，并结合LLM生成的关键词描述，模型能够学习关键词之间的语义关系，实现高效、精准的关键词推荐，显著提升卖家满意度和广告效果。

AB：线上AB实验中，关键词推荐的点击率提升了2.3%，卖家满意度提升了5.1%。

心得：

（1）图神经网络与大语言模型的结合是一个非常有前景的方向。

（2）关键词推荐是一个非常实际的问题，值得深入研究。

可信度：生产——eBay

推荐等级：值得看看
