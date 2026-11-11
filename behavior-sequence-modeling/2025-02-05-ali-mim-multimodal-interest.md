# 推荐算法(2025.2.4)：阿里MIM，多模态内容兴趣建模用于行为序列

> 原文：https://mp.weixin.qq.com/s/0-a1tBUDlUzowfI-MuDTBA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-05（标题日期 2025.2.4）
> 本文拆自《推荐算法paper（2025.2.4）》，同期 2 篇：[阿里 MIM 多模态兴趣建模](../behavior-sequence-modeling/2025-02-05-ali-mim-multimodal-interest.md)、[eBay 广告关键词中间人偏差](../deep-retrieval/2025-02-05-ebay-middleman-bias.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-05-ali-mim-multimodal-interest.full.en.md)

MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling

网址：https://arxiv.org/pdf/2502.00321

公司：阿里巴巴

方向：多模态+行为序列

解读：将多模态用于行为序列建模。多模态模型，在用于行为序列建模前，要在自己的业务数据上做预训练和对齐工作：即将一个好用的公网预训练模型（cv、nlp、两者兼具等的），再在自己的业务上做预训练，实现图与文本的对齐、query与candidate的对齐等。

多个工作已经表明这一点，candidate item的多模态embedding要放入推荐模型前，要做充分的预训练和各种对齐，特别是与id embed的对齐。不推荐的做法是直接塞入。

可信度：生产

推荐等级：值得看看
