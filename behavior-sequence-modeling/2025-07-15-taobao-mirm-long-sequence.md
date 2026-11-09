# 推荐算法(2025.7.15)：淘宝的长序列召回新方法GMV增长4个点

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484890&idx=1&sn=a0f75d4b0b5396e9efb6ad35af31f804&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-07-15

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-07-15-taobao-mirm-long-sequence.full.en.md)

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 MIRM: Towards the Modeling of Interest Retrieval and Matching for Long-term User Behavior Sequence

网址：https://arxiv.org/pdf/2507.10714

公司：淘宝

方向：长序列召回

解读：

背景：传统推荐系统依赖近期行为或通过简单扩展序列长度处理长行为序列，但受到计算成本和记忆限制。长序列召回是解决工业中长序列问题的方向，例如用户行为序列长度通常长达数万，如何高效提取兴趣并精准检索相关item。

核心思想：MIRM框架旨在解决长序列召回的效率和准确性问题。该方法通过将长序列分解为多个子序列，并利用多种检索策略，提取用户长期兴趣，从而提高推荐系统的召回效果。具体包括：

（1）使用MIRM模型进行兴趣建模和匹配。长序列召回系统通常由兴趣建模和兴趣匹配两个组件组成。MIRM首先将用户长行为序列分解为多个子序列，基于行为类别和时间间隔。然后，通过检索策略（包括最近邻和最大内积搜索），提取目标用户的兴趣信息，并与候选item进行匹配。

（1）用近期行为序列的pooling结果作为query，从上面说的各个子序列里挑选出与target item同category的子序列做Target Attention，生成对应的embedding，作为长序列兴趣embedding。注意，inference的时候，要与top-k个子序列生成多个兴趣embedding。与此同时，近期行为序列通过self-attention获得一个短兴趣embedding。

两类兴趣向量，连同target item等，计算user和target item的match概率，并分别通过两个Loss加权和，进而优化整个模型。注意，样本里的target item都是用户互动过的正样本，从target item同类别sampling出来一些负样本，只用于计算match概率的时候做分母用途。

（2）使用Pointer-Generator Interest Network (PGIN)预测顶级K兴趣类别。基于顶级K类别，进行并行ANN搜索：长兴趣嵌入仅从对应类别检索，短兴趣嵌入从全候选池检索，确保离线一致性，减少计算开销。

AB贴一下：在淘宝秒杀小程序上运行三周，提升5.54%点击、11.01%订单、4.03% GMV。整体系统RT增加约15ms。

心得：（1）长序列召回是推荐系统中处理用户长行为序列的重要方向。MIRM框架通过将长序列分解为多个子序列，并结合多种检索策略，实现了高效的兴趣建模和匹配。（2）这篇文章中的长兴趣提取方法在淘宝这样的平台上表现出了良好的效果，同时PGIN的设计也为并行ANN搜索提供了有效的解决方案。

可信度：生产

推荐等级：推荐看看

<!-- 本文配图：页面图片元素未返回图片链接，未能存档 -->

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.7.10)：淘宝的意图建模纠偏方法AB效果两位数](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484884&idx=1&sn=8a0e3c23281188cab8a4660ece028c23&scene=21#wechat_redirect)

■ [推荐算法(2025.7.8)：华为的特征选择的新方法](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484878&idx=1&sn=d1e5248e2caad9e02911c41eac83df8f&scene=21#wechat_redirect)

■ [推荐算法(2025.7.3)：快看Tiktok是怎么审核你的视频的](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484873&idx=1&sn=cfd6a674121ce40b7de334885b4d4b80&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
