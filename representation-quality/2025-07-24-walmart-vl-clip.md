# 推荐算法(2025.7.24)：沃尔玛，抠啊抠，抠出GMV大涨10个点

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484923&idx=1&sn=09efe810597012d381d8efa1bfa56ded&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-07-24（仅自己可见）

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings

网址：https://arxiv.org/pdf/2507.17080

公司：沃尔玛

方向：搜索+相似推荐+LLM

解读：

（背景）传统CLIP模型在电子商务中面临挑战，如产品图像背景杂乱（生活场景）和文本元数据冗余（结构化和非结构化信息），导致推荐准确率低。提出一个框架，分别通过CV模型和大语言模型去除了item的图像、文本里的noise。包括两个关键组件：

（1）视觉基础

使用Grounding DINO模型检测并裁剪图像中的主产品，移除背景干扰，生成更纯净的视觉表示。

Grounding DINO模型是一个CV模型，能从一张图提取指定物品（列表）的图像元素。

两个输入——产品的一些meta信息作为指定物品（列表）和商品的商品图像，就能提取出来图像里的主体。这样做的好处，将无关的物品以及背景去除了，屏蔽了图像里的noise，这是本paper最大的创新点。以往工作都是将整个图像跟相应的text对齐。

（2）LLM增强文本表示

采用大型语言模型（LLM，如GPT-4）通过总结-评估-精炼的三步链式过程，生成简洁、视觉导向的文本描述（限20词以内），聚焦颜色、纹理、形状和材料等可观察属性。这样做，可以去除了文本（描述和标题）里的noise。

模型架构：基于CLIP的对比学习框架，训练图像和文本嵌入的相似性，使用HNSW索引实现高效检索。

可以用于两个业务场景：

（1）搜索，用query的embed搜索商品的图像的top embed。

（2）相似推荐，通过query product的图像的embed，搜索相似embed的商品。

AB：在沃尔玛网站上，相比基线模型，点击率（CTR）提升18.6%、加购物车率（ATC）提升15.5%、总销售额（GMV）提升9.9%。没有说具体是哪个业务，还是这两个业务的总和。

心得：

（1）Grounding DINO模型抠图去除noise真是一个神奇的操作。业界，Grounding DINO模型是一个被证明很有效的模型。

（2）效果提高那么多，真是很令人吃惊。图里的noise真是那么大吗？如果是，以往很多大厂没有拿到效果或者效果较小，可能这就是原因。

可信度：生产

推荐等级：有实践价值

![](https://mmbiz.qpic.cn/mmbiz_png/6JoibB6eg3um6U7ylKwMick4PibiazkHBz4icybY3p2wYnzMU1ibpSpmujlVw8P9lZYrTndiaJWFFVO6wKfOF4J1P6Qw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1)

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.7.23)：Meta的解藕实践取得很大成绩](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484917&idx=1&sn=7825158d39731034eb3945c1888a9fe4&scene=21#wechat_redirect)

■ [推荐算法(2025.7.22)：字节的推荐算法“大模型”有效果](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484908&idx=1&sn=7015e595144e0712fb4401175f50e624&scene=21#wechat_redirect)

■ [推荐算法(2025.7.19)：Pinterest出品必是精品——GPT建模行为序列](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484902&idx=1&sn=2ebdcda5dceefb8d2608f48bdeaab839&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

请入群，群内可讨论技术方案、发招聘广告、内推和交友。

• 铁粉标准：累计10次评论或爱心、投稿1次。

• 入群方法：请您加个人微信lmxhappy，我拉您入群，请备注（只我个人看，不公开）。

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
