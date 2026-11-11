# 推荐算法paper清单（2024.11.19）

> 原文：https://mp.weixin.qq.com/s/xNpSP1hhr5jJw3R_XegXdg
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-19

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-11-19-four-papers.full.en.md)

1 QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou

网址：https://arxiv.org/pdf/2411.11739

公司：快手。周国瑞团队。

方向：rank+多模态

概述：将多模态的embed做转换后，再用于排序模型。已有工作存在的两个问题：一是cv/nlp等的模型与排序模型的训练目标不相同，二是多模态的embed不可学习。该工作就是为了缓解这两个问题。

效果：生产

推荐等级：值得看看

2 Collaborative Contrastive Network for Click-Through Rate Prediction

网址：https://arxiv.org/pdf/2411.11508

阿里淘宝。

方向：rank+后推荐（TIR）

概述：通过对比学习，构建trigger item与target item的关系。这是TIR里，第一次引入对比学习。在淘宝在线A/B测试，CTR提升了12.3%，订单量增加了12.7%，由此对比学习的强大能力。

效果：生产

推荐等级：值得看看

3 All-domain Moveline Evolution Network for Click-Through Rate Prediction

网址：https://arxiv.org/pdf/2411.11502

公司：阿里淘宝。跟上一篇paper作者相同，一天发布两篇paper，没天理啊。

方向：rank+意图识别

概述：以往的意图识别都是item级别的，单场景级别的，即在CTR模型里，用户对target item（或者target item）的意图建模，且是利用的单个场景的数据建模的。本paper希望将用户在所有场景的数据收集起来，构建用户的真实用途。这个idea是一个很naive的。因为单场景是很难获取用户的意图的。只有将他在app上的所有用户行为串起来，才能更好的理解他的意图，才能更好的为其推荐。

效果：生产

推荐等级：值得看看

4 Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning

网址：https://arxiv.org/pdf/2411.11225

公司：快手

方向：rank+冷启动

概述：能力有限，花了十分钟没看懂，god。

效果：生产

推荐等级：有时间看看。做冷启动的有时间看看。
