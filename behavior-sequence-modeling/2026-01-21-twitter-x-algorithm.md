# 推荐算法(2026.1.21)：推特的推荐模型特别简单易懂，3分钟读懂

> 原文：https://mp.weixin.qq.com/s/3tvsymPAHEPUzAT1IXj9dQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2026-01-21（修改于2026年1月22日）

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2026-01-21-twitter-x-algorithm.full.en.md)

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s/BhVGu0qH5kTz9N5zpsD3KA)

1 开源模型https://github.com/xai-org/x-algorithm

公司：XAI

思想：端到端

方向：Transformer排序

解读：

本代码库是推特推荐频道的推荐流程，涉及到召回、排序和重排。没有粗排环节。

（1）召回

多路召回，比如：

1）基于内容相似度的召回（emb召回）；

2）基于知识图谱的召回；

3）基于协同过滤的召回；

4）基于热门的召回。

（2）排序模型

Grok-1推荐版：在Grok-1基础上修改了attention范围，它是一个transformer模型，它实现了平台所有行为相关的序列建模。特别的，

1）用户行为序列、社交关系、帖子内容、实时上下文等特征都作为token输入到同一个transformer中，统一建模；

2）模型没有采用传统的“用户塔+物料塔”结构，而是直接对用户和候选物料的联合序列进行建模；

3）**多用户、多candidate的**推理：为一个用户召回的所有candidate item作为一个序列输入，模型做attention的时候，candidate互相之间不做attend，从而互不影响。也就是说利用了transformer的序列建模能力，但是实现了独立的target attention的效果。

其中1）、3）是亮点，特别是3）。2）是常规操作。

（3）重排

1）多样性重排；

2）新鲜度重排；

3）平台内容政策合规性检查。

心得：

（1）多用户的、多candidate在一个batch里推理，将低效的Transformer高效化，从而落地生产环境。

（2）极致的端对端。所有特征都不手动生产，召回只有ebr召回。

（3）对平台内容质量有非常高的要求

召回、重排相对来说，算是非常简单了。这就要求平台上的物料质量是非常高的。而物料库是百万级，我估计对物料做了大量的过滤筛选，从而保留了很多高质量的物料。

可信度：生产

推荐等级：有实践价值

请帮忙点赞、转发，谢谢。

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

[Meta、快手、Shopee…生成式推荐路线已清晰，你跟不跟？](https://mp.weixin.qq.com/s/uWRy2fgknJjSGOwTLeA8Rw)

[2025年搜广推论文投票](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247485398&idx=1&sn=5aefda400eae438daa8078711c7bfa93&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

请入微信群，群内我会给出更深入的解读，还可以共同讨论技术方案、发招聘广告、内推和交友等。

• 铁粉标准：关注公众号一个月以上，且在公众号上累计15次互动（评论、爱心、转发）、或投稿1次，只欢迎技术同学。

• 入群方法：请您加个人微信lmxhappy，我拉您入群，请备注（只我个人看，不公开）。

喜欢的话点个在看吧👇

个人观点，仅供参考

修改于2026年1月22日
