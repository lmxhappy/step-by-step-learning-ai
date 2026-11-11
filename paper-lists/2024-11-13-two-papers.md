# 推荐算法paper清单（2024.11.13）

> 原文：https://mp.weixin.qq.com/s/KN41e96_U3XlyU0PtFNJRQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-12（修改于2024年11月12日）

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-13-two-papers.full.en.md)

1.Towards Automated Model Design on Recommender Systems

网址：https://arxiv.org/pdf/2411.07569

公司：meta

方向：推荐系统优化+AutoML。

概括：提出了一种范式，实现了模型效能的提高。FLOPs效率上提高了2倍，能效提高了1.8倍，性能提高了1.5倍。

可信度：生产

推荐等级：有时间看看。meta那样的超大公司，才对性能有更加急迫解决的压力和动力。

2.AdaS&S: a One-Shot Supernet Approach for Automatic Embedding Size Search in Deep Recommender System

网址：https://arxiv.org/pdf/2411.07504

公司：腾讯

方向：AutoML。

概括：搜索动态的特征的embedding size。我们将主模型训练与搜索过程解耦，这确保了参数的合理性和嵌入尺寸分配的稳定性。为训练超级网络设计了自适应采样方法，极大地提高了其一致性。在搜索阶段采用基于强化学习（RL）的方法来寻找最优的AES结果。

可信度：离线验证

推荐等级：有时间看看。同样的，一般公司也用不上，也玩不了。
