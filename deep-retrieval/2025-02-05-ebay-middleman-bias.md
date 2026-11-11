# 推荐算法(2025.2.4)：eBay广告关键词推荐的中间人偏差，相关性要与搜索对齐

> 原文：https://mp.weixin.qq.com/s/0-a1tBUDlUzowfI-MuDTBA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-05（标题日期 2025.2.4）
> 本文拆自《推荐算法paper（2025.2.4）》，同期 2 篇：[阿里 MIM 多模态兴趣建模](../behavior-sequence-modeling/2025-02-05-ali-mim-multimodal-interest.md)、[eBay 广告关键词中间人偏差](../deep-retrieval/2025-02-05-ebay-middleman-bias.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-05-ebay-middleman-bias.full.en.md)

Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search

网址：https://arxiv.org/pdf/2502.00131

公司：ebay

方向：广告搜索+关键词推荐

解读：这篇论文主要探讨 eBay 电商平台广告关键词推荐相关性问题。研究发现基于点击 / 销售信号训练相关性过滤模型存在缺陷，因为数据存在偏差，且搜索排名影响点击行为。文中提出 "中间人偏差" 概念，强调需让广告关键词与搜索相关性信号一致。研究对比了双编码器和交叉编码器在模拟这种一致性方面的性能及可扩展性，结果表明简单的两层交叉编码器在相关性建模上表现更优，虽双编码器速度快，但轻量级交叉编码器能缩小延迟差距并带来更好效果。

可信度：生产

推荐等级：有时间看看
