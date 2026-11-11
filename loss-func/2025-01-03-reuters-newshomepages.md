# 推荐算法(2025.1.3)：路透社用新闻首页版面学习新闻重要性排序

> 原文：https://mp.weixin.qq.com/s/kx6LdmgPOSOOGQYsyiME_Q
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-03
> 本文拆自《推荐算法paper（2025.1.3）》，同期 2 篇：[快手 S-Diff 频谱域扩散协同过滤](../graph-recommendation/2025-01-03-kuaishou-s-diff.md)、[路透社 NewsHomepages 新闻重要性](../loss-func/2025-01-03-reuters-newshomepages.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-03-reuters-newshomepages.full.en.md)

NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions

网址：https://arxiv.org/pdf/2501.00004

公司：路透社

方向：新闻重要性判断+pair-wise

解读：

收集了超过3000个新闻网站首页的快照，根据文章在首页上的位置和大小，形成pair样本，然后用基于Transformer的二元分类器，来预测文章的相对重要性。进一步，两个下游任务：一是分析不同新闻机构之间新闻价值判断的相似性；二是评估新闻的新闻价值。可以看出，路透社对全世界各个领域新闻动态的检测是多么的认真和努力，也说明了他们新闻权威性的来源——科技创新。

可信度：生产

推荐等级：有时间看看
