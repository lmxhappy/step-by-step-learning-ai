# 推荐算法(2025.7.16)：Meta的相关短视频推荐的相关性提升了12%

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484896&idx=1&sn=00e46baa7b3bedc78520a72f4360ee74&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-07-16

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-07-16-meta-related-video-relevance.full.en.md)

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 Balancing Semantic Relevance and Engagement in Related Video Recommendations

网址：https://arxiv.org/pdf/2507.09403

公司：Meta

方向：双塔召回+i2i

解读：

双塔召回的时候，通过增加一个辅助任务，增加candidate item与trigger item的内容相关性。实际上，trigger item和candidate item用的是孪生网络（就是共用一个encoding网络）。

背景和动机：

传统协作过滤（CF）基于共同参与信号（如共同观看），但存在局限：语义模糊：噪声互动导致无关视频关联；流行度偏差：热门内容主导，减少多样性。缺乏可解释性：纯行为模型不透明。

具体方法：

除了参与度任务，增加了一个相关性任务。两个任务共用同样的双塔输出的embed，辅助任务只是在参与任务的Loss上，增加了一个开关，即只有语义相关且共同参与的，开关是开的，即Loss有效，否则，开关是关的，即Loss被乘以0，即0 Loss了。

主任务和辅助任务的Loss加权求和，联合训练，优化模型。

本方法成功的关键——trigger item塔引入上下文，提升了建模效果：

在双塔里，一个塔对应trigger item，一个塔对应candidate item。在tirgger item塔里，不仅要考虑trigger item还考虑了前序的最近的用户行为序列（K个用户互动item）。最终的embedding将context序列与trigger item的embed concat一起，并通过dense层获得一个最终的embed（与candidate item同长度）。

除了辅助任务，还有两个trick：

（1）多模态内容特征（文本和视觉嵌入）的融合，以实现更丰富的语义理解；

（2）通过逆倾向加权的离线策略修正（OPC），有效缓解流行度偏差。

贴AB：facebook短视频推荐，语义相关性从51%升至63%，热门视频分布降13.8%，顶级参与度指标升0.04%，短期参与略降但长期保留改善。

心得：

（1）将多个信号作为Label是多年以来的方法，在主任务上增加一个Loss开关并作为辅助任务，也不鲜见，但是也不常见，perfect。特别是在item-to-item任务里，将相关性信号注入到label里，真是很有效的。

（2）将多个信号作为Label是多年以来的方法，在主任务上增加一个Loss开关并作为辅助任务，也不鲜见，但是也不常见，perfect。特别是在item-to-item任务里，将相关性信号注入到label或者loss里，是最近的常见做法。

可信度：生产

推荐等级：有实践价值

<!-- 本文配图：页面图片元素仅返回占位图，图片链接未能取到 -->

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.7.15)：淘宝的长序列召回新方法GMV增长4个点](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484890&idx=1&sn=a0f75d4b0b5396e9efb6ad35af31f804&scene=21#wechat_redirect)

■ [推荐算法(2025.7.10)：淘宝的意图建模纠偏方法AB效果两位数](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484884&idx=1&sn=8a0e3c23281188cab8a4660ece028c23&scene=21#wechat_redirect)

■ [推荐算法(2025.7.8)：华为的特征选择的新方法](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484878&idx=1&sn=d1e5248e2caad9e02911c41eac83df8f&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
