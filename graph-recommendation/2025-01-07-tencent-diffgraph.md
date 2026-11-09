# 推荐算法paper

> 原文：https://mp.weixin.qq.com/s/X11XVmPninN2X_EbOmxlYQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-07

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-07-tencent-diffgraph.full.en.md)

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

1 DiffGraph: Heterogeneous Graph Diffusion Model

网址：https://arxiv.org/pdf/2501.02313

公司：腾讯

方向：GNN+diffusion

解读：利用diffusion的强大祛噪能力，帮助GNN建模过程中的祛噪。不是直接在原始图结构上添加和去除噪声，而是在图的潜在表示空间中进行扩散。

可信度：离线数据

推荐等级：有时间看看

请帮忙点赞、转发，谢谢。

个人观点，仅供参考
