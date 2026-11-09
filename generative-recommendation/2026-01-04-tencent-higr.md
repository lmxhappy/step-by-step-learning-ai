# 推荐算法(2026.1.5)：腾讯，一站式推荐，观看时间+1.22%

> 原文：https://mp.weixin.qq.com/s/fTek0UvLVEHcYuHMRszhiA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2026-01-04（标题已修改；修改于2026年1月7日）

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2026-01-04-tencent-higr.full.en.md)

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s/BhVGu0qH5kTz9N5zpsD3KA)

1 HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment

网址：https://arxiv.org/pdf/2512.24787

公司：腾讯

思想：生成推荐生成一次的整个推荐列表

方向：生成推荐+listwise+一站式推荐

解读：

提出了一个生成整个推荐列表的生成模型，主要由生成SID、生成模型和后训练组成。

（1）生成SID——对比 RQ-VAE (CRQ-VAE)

为了解决下面几个问题：

1）RQ-VAE得到的SID的token的语义不明确，存在语义鸿沟；

2）RQ-VAE中，相同item可能被映射到不同的SID（缺乏唯一性）；

3）RQ-VAE中，不同item可能被映射到相同的SID（缺乏区分性）。

因此，提出了Contrastive RQ-VAE（CRQ-VAE），主要包括：

1）重建损失：保留原有RQ-VAE的重建损失，以确保SID的语义信息；

2）对比损失：通过对比学习，将相同item的SID拉近，不同item的SID推远，从而提高SID的唯一性和区分性；

3）多样性损失：通过鼓励不同item的SID之间的差异，提高SID的多样性。

（2）生成模型

通过堆叠三个变换，逐步的将用户行为序列，变成最终的推荐列表。

1）上下文

将用户行为序列的item的SID和item的特征（如类目、价格、品牌等）拼接成一个向量，并通过一个MLP获得一个向量，作为上下文的表征。

2）粗粒度层

将上下文的表征，通过一个transformer层，获得一个粗粒度的推荐列表的表征。

并且堆叠这些变换层

注意，每个变换都要做resnet的短接。

3）细粒度层

将粗粒度层的输出，通过一个transformer层，获得一个细粒度的推荐列表的表征。

对预测结果计算NTP loss，优化模型。

（3）后训练

用监督的对比学习对后训练，对齐用户的偏好。二分类，构建偏好对：正样本 为用户真实互动序列（按反馈降序重排）；负样本包括随机置换（优化排名）、负反馈替换（优化兴趣）、重复锚点附加（优化多样性）。这确保生成整体优化排名、兴趣和多样性。

（4）预估

用生成模型，自回归预测M个item。预测过程中使用beam search。通过反向查找码本对应的item确定item。

AB：视频号，在 2% 流量上，提升平均停留时间 1.03%、观看时间 1.22%、视频浏览 1.73%、请求数 1.57%

补充：

RQ-VAE存在的问题：

1）ID 纠缠与稀疏：RQ-VAE的SID token存在语义纠缠，不同item的SID token可能存在重叠，导致模型难以区分不同item。

2）缺乏协作对齐：RQ-VAE的SID token缺乏与用户行为的协作对齐，导致模型难以学习到用户行为和item之间的关系。

3）低效多样性控制：RQ-VAE的SID token缺乏多样性控制，导致生成的推荐列表可能存在重复或相似的item。

心得：

（1）listwise少有成功的实现，而生成式推荐能够输出多个item，可以将其作为listwise实现的一种方法。

愚见：

公式（10）有笔误

可信度：生产

推荐等级：有实践价值

请帮忙点赞、转发，谢谢。

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

[推荐算法(2025.12.23)：Pinterest，物料冷启动，互动10%](https://mp.weixin.qq.com/s/TEaQPAsJkMWsy0Yp8CeNAQ)

[推荐算法(2025.12.12)：阿里，挖掘其它域的增量样本，大幅提升召回效果，交易量提升1.2%](https://mp.weixin.qq.com/s/VEse6yYsvYQfNR0YOCVeXg)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

请入微信群，群内我会给出更深入的解读，还可以共同讨论技术方案、发招聘广告、内推和交友等。

• 铁粉标准：关注公众号一个月以上，且在公众号上累计15次互动（评论、爱心、转发）、或投稿1次，只欢迎技术同学。

• 入群方法：请您加个人微信lmxhappy，我拉您入群，请备注（只我个人看，不公开）。

喜欢的话点个在看吧👇

个人观点，仅供参考

修改于2026年1月7日
