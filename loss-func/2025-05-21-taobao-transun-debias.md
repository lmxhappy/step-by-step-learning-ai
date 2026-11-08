# 推荐算法(2025.5.21)：淘宝的回归预测偏差debias

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484681&idx=1&sn=24323cbd26f6208cedb928fd8feedc87&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-05-21（标题已修改；修改于 2025-05-21）

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

📄 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models in Recommender Systems

网址：https://arxiv.org/pdf/2505.13881

公司：阿里淘宝

方向：debias（重变换偏差）

解读：

为了解决MSE回归模型的预测偏差，本文在已有变换MSE模型中，增加一个debias网络。

实际目标是y，最终的预测结果是y'，但是预测网络预测的是f(x)，它的label是转换函数（如对数），y'与f(x)的关系是f(x)经过反向转换函数（如指数）。

本文的方法是除了还要继续训练f(x)，还要同时训练一个debias网络z(x)，它的目标是y/y'。别忘记y'与f(x)的关系，仅仅是一个转换的关系。训练完，inference的时候，只要y'*z(x)作为y的预测结果就行了。这是因为y/y'=z(x)，将y'移到右侧，就是y=y'*z(x)。这也就是作者说的理论上保证了这种方法是无偏的。

什么是重变换偏差？重变换偏差（retransformation bias）是指在回归模型中使用目标变换（如对数、平方根等）来处理非正态分布数据时，模型预测值在通过逆变换（retransformation）回到原始目标空间后，偏离真实期望值的问题。

心得：

（1）看似没有创新的地方仍然可以创新，我看到了第一性原理在发挥作用，也看到了作者的辛勤付出。

（2）有bias，就用debias网络就debias。

可信度：生产——淘宝首页

推荐等级：推荐看看

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法周报(2025.5.17)：易懂的大厂方法更值得复现](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484676&idx=1&sn=152d7ee5765924a325a9fd0d8e7bccdb&scene=21#wechat_redirect)

■ [推荐算法(2025.5.16)：Airbnb的搜索Rerank全量模型](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484666&idx=1&sn=86cad0ea9efa3094e18303eee515c905&scene=21#wechat_redirect)

■ [推荐算法(2025.5.15)：Google的新物料流量分配机制](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484660&idx=1&sn=af9370e1a5b04b9996b5ad0d4a7ecd56&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容，请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考

修改于2025年5月21日
