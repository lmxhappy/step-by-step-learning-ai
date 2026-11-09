# 推荐算法(2025.7.8)：华为的特征选择的新方法

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484878&idx=1&sn=d1e5248e2caad9e02911c41eac83df8f&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-07-08

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-07-08-huawei-tayfcs.full.en.md)

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems

网址：https://arxiv.org/pdf/2507.03895

公司：华为

方向：特征选择

解读：

一种用于特征组合选择的方法，核心目标是从大量可能的特征组合中筛选出重要且非冗余的特征组合，剔除不重要的或冗余的特征组合。具体包括：

（1）TayScorer 模块

基于泰勒展开分析特征对模型预测的贡献，通过信息矩阵等式（IME）近似计算高阶特征组合的重要性。无需预构造所有特征组合，仅需一次反向传播即可高效估计特征重要性，时间复杂度从 O(n²) 降至 O(n)。选择特征嵌入的均值作为泰勒展开的起点（x0），确保评估的公平性，并通过梅度外积近似高阶导数。

（2）LRE（Logistic Regression Elimination）模块

使用逻辑回归作为代理模型，通过滑动窗口逐批添加特征组合，评估其信息增益（基于验证集性能）。若特征组合带来负增益（即冗余），则移除。有效消除特征组合间的冗余，确保选择的特征组合信息量最大化。

（3）嵌入表优化

通过哈希技术压缩特征组合的嵌入表，将超过阈值（𝜏 = 5.0e6）的特征值映射到固定大小的索引，降低内存开销。

贴AB：在广告平台 CVR 预测任务中，提升了 13.9% 的转化率和 0.73% 的收入。

可信度：生产

推荐等级：推荐看看

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.7.3)：快看Tiktok是怎么审核你的视频的](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484873&idx=1&sn=cfd6a674121ce40b7de334885b4d4b80&scene=21#wechat_redirect)

■ [推荐算法(2025.7.1)：Pinterest的兴趣聚类召回](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484867&idx=1&sn=33c9c879ca6f9282601c2709750f69dc&scene=21#wechat_redirect)

■ [推荐算法(2025.6.30)：DCN模型的第三版效果不错](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484859&idx=1&sn=6c06a6dcb3c5eddb2aa4d8b9dca8f2d&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
