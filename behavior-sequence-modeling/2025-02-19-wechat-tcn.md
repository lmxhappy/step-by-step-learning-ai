# 推荐算法paper速看！微信在序列建模上的新突破

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484159&idx=1&sn=011e9ad100db2741e88da559d8c7655b&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-19

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-19-wechat-tcn.full.en.md)

[本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 Introducing Context Information in Lifelong Sequential Modeling using Temporal Convolutional Networks

网址：https://arxiv.org/pdf/2502.12634

公司：微信频道

方向：序列建模

解读：行为序列建模是推荐领域的一个重要方向，微信的这个工作，在使用TCN建模行为序列的基础上，引入上下文信息，即给user的序列，计算相邻item上下文感知的的embedding，来提高模型效果。

（1）背景：行为序列建模的关键是注意力模块，它从序列中得出与候选项目相关的兴趣表示。通常，注意力模块以逐点的方式运行，只关注序列中单个项目与候选项目的相关性。然而，相邻项目中有助于更准确评估每个项目重要性的上下文信息却未被考虑在内。

（2）方法：在计算某一个sequence item的embedding的时候，将前后一定程度的item，组成一个定长的序列，通过一个线性层，获得相应的embedding。

除此之后，还用到了dynamic networks等trick。

心得：行为序列建模是一个热点方向，有大量的paper，且给出一些有启发性的idea，感觉还可以大有可为，可以做的工作还有很多，如多行为、上下文感知、长序列、引入其它关系等，同仁们加油啦。

可信度：生产

推荐等级：推荐看看。

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

推荐您继续阅读：

■ [推荐算法paper必看！推荐界首次深度借鉴 DeepSeek，开启创新探索之旅](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484154&idx=1&sn=8f21bd9302b484f856855241759ca345&scene=21#wechat_redirect)

■ [推荐算法周报：华为、阿里、快手等大厂最新研究成果，助你领先一步！](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484145&idx=1&sn=55e812648c7dd4342ba750b808b7d9d1&scene=21#wechat_redirect)

■ [推荐算法paper华为序列位置编码最新研究成果，助你领先一步！](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484136&idx=1&sn=4cdcc457302756a2be8aec0eb3fd5014&scene=21#wechat_redirect)

个人观点，仅供参考
