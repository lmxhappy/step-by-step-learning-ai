# 推荐算法(2025.4.15)：Meta URM，一个模型统一召回、排序等推荐任务

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484486&idx=1&sn=6687984fa2dd4d718ea0c55554f2747d&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-15（标题已修改；修改于 2025-04-15）
> 本文拆自《推荐算法(2025.4.15)：3篇大厂非大模型相关的论文》，同期 3 篇：[Meta URM 统一推荐模型](../deep-retrieval/2025-04-15-meta-urm-unified-model.md)、[Meta FROG 多模态好友推荐](../graph-recommendation/2025-04-15-meta-frog-friend-rec.md)、[Meta 无监督数据增强序列推荐](../behavior-sequence-modeling/2025-04-15-meta-unsupervised-seq-enhancement.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-04-15-meta-urm-unified-model.full.en.md)

Unified Recommendation Model (URM): Retrieval, Ranking and Beyond

网址：https://arxiv.org/pdf/2504.09631

公司：Meta

方向：召回

解读：

提出一种统一的推荐模型，能够同时完成检索、排序以及其他推荐任务。核心在于通过统一的输入表征和Transformer结构来建模。

（1）输入表征：每个用户：用户侧信息用一个[U] token描述（可以包含用户特征、上下文等）以及多个候选项目，每个候选项目对应一个[C] token。模型通过处理这些token序列来生成用户的点击概率。

（2）编码器/生成器：编码器和生成器都是Transformer。编码器处理完整的token序列，生成上下文相关的嵌入表示，生成器根据这些嵌入生成目标项目token的点击概率。生成器采用自回归方式，基于已生成的token序列，预测下一个项目的点击概率。

（3）多目标推荐：同时支持多样化的推荐目标（如多样性、个性化），通过不同的目标token生成相应项目的推荐概率。目标token为[U]、[C]或者其他特定目标。每个目标对应不同的token（如“推荐相似项目”或“推荐多样化项目”），生成对应的项目。

心得：统一模型，解决跨任务迁移的问题，是一个很好的方向。

可信度：离线

推荐等级：推荐看看
