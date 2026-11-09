# 推荐算法(2025.9.19)：Snap的序列样本的数据增强方法，有实践价值

> 原文：https://mp.weixin.qq.com/s/gpTJFQA2WLBYsVFDlOw5TA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-09-19

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-09-19-snap-seq-augmentation.full.en.md)

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s/BhVGu0qH5kTz9N5zpsD3KA)

1 Sequential Data Augmentation for Generative Recommendation

网址：https://arxiv.org/pdf/2509.13648

网址：Snap

方向：行为序列+生成式推荐+数据增强

解读：

为生成式推荐模型的样本做数据增强，以解决其样本稀疏的问题，特别是对于其中的长尾样本。即通过采样子序列、target item，生成一个样本，用于训练生成式推荐模型。

采样过程：

（1）序列采样：从所有用户序列中采样一个序列u。这是采样user。

（2）目标采样：在选定序列中采样位置k作为next item（目标y）。

（3）输入采样：在target item前采样子序列作为输入x。这是从一个完成序列里，采样一个子序列。

三种采样方法（序列采样、目标采样、输入采样）顺序组合使用，生成训练对（行为序列，target item），用于生成式推荐模型的训练。

补充背景知识——生成式推荐模型样本的构建策略，简化成三种策略：

（1）Last-Target (LT)：每个用户只生成一对：输入为序列除最后一件，目标为最后一件。简单，但忽略序列内部监督信号。

（2）Multi-Target (MT)：每位位置k≥2生成一对：输入为前k-1件，目标为第k件。提供更多信号，但输入固定为完整前缀。

（3）Slide-Window (SW)：更激进，对每个目标kkk，枚举所有可能窗口大小的输入子序列。多样性高，但计算昂贵，可能引入噪声。

心得：

（1）数据增强是一个常用的手段。一个新方向出现，可以考虑这个手段，实现创新。

愚见：行为序列建模也可以试试这个数据增强方法。

可信度：生产

推荐等级：有实践价值

请帮忙点赞、转发，谢谢。

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法paper](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484024&idx=1&sn=b6c0bc04592d62da52ce1cfe36e84d1a&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

请入微信群，群内我会给出更深入的解读，还可以共同讨论技术方案、发招聘广告、内推和交友等。

• 铁粉标准：关注公众号一个月以上，且在公众号上累计10次互动（评论、爱心、转发）、或投稿1次。

• 入群方法：请您加个人微信lmxhappy，我拉您入群，请备注（只我个人看，不公开）。

喜欢的话点个在看吧👇

个人观点，仅供参考
