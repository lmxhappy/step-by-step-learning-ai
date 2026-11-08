# 推荐算法paper

> 原文：https://mp.weixin.qq.com/s/QnuAOHmIElrdkCSQFs7mQg
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-05（标题日期 2025.1.30）

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

* 推荐标准：大都是大厂的、少量学术的ranking model。
* 更新频率：日更。
* 来源：arxiv.org。
* 阅读建议：优先看前面的。

```
1 推荐等级：现在只有两个——推荐看看，有时间看看。偶尔会"强烈推荐看看"。
2 * 推荐看看，做ranking的同学需要及时了解一下，对于工作有帮助。
3 * 有时间看看，如名。
4
5 可信度：包括离线和生产。
6 * 生产：能用到生产中的方法，有用性会相对较大，他人更原因用在自己的业务上。
7 * 离线：相比，离线会小一些。
```

1 Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation

网址：https://arxiv.org/pdf/2501.17670

公司：快手

方向：Vector Quantization（实时聚类）+行为序列+Diffusion

解读：利用Vector Quantization（实时聚类）和Diffusion等两种技术，提升行为序列的用户兴趣建模。（1）Vector Quantization：就是通过实时聚类，维护一个codebook，其中保存了cluster centroids。（2）Diffusion：扩散模型的前向过程是逐渐向真实的下一个物品表示中添加噪声，而反向过程则是在用户交互序列引导下从噪声中逐步恢复，去噪的目标是得到准确的下一个物品表示。

不禁感叹kmeans这么牛，又一篇相关的paper。聚类提升效果，已经在行为序列、user表征、item表征等上，被多个公司证明有效。读者大人们，你们也考虑一下应用到你们的场景，最好发paper出来哈。

很好奇，作者名单里，为什么没有带盖坤、周国瑞的大名。

可信度：离线

推荐等级：推荐看看

请帮忙点赞、转发，谢谢。

个人观点，仅供参考
