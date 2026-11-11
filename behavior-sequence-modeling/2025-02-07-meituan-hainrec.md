# 推荐算法(2025.2.7)：美团HAINRec分层注意力交互序列推荐

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484080&idx=1&sn=f2dedc72ef006b9239eb32fc94dafc3d&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-07
> 本文拆自《推荐算法paper（2025.2.7）》第 2 篇。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-07-meituan-hainrec.full.en.md)

HAINRec: Hierarchical Attention Interaction Network for Sequential Recommendation

网址：https://arxiv.org/pdf/2501.19158

公司：美团

方向：序列推荐

解读：美团关于序列推荐的最新paper。HAINRec提出了一种分层注意力交互网络，通过将序列建模与注意力机制结合，实现了高效的序列建模。具体来说，HAINRec首先将用户行为序列转换为嵌入向量，然后通过分层注意力机制，捕捉序列中不同层次的兴趣变化。同时，HAINRec还引入了一种新的交互模块，通过将序列建模与注意力机制结合，实现了高效的序列建模。

可信度：离线

推荐等级：值得看看
