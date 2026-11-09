# 推荐算法paper

> 原文：https://mp.weixin.qq.com/s/HxzD5DwkOTXc-d2sqtz87A
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-16

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-16-bytedance-realtime-indexing.full.en.md)

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

1 Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever

网址：https://arxiv.org/pdf/2501.08695

公司：字节

方向：深度召回

解读：不用ANN方式召回，而是通过两步实现了实时深度召回——（1）实时kmeans聚类、归类，并（2）通过DNN对item最近似的cluster的所有item进行ranking。

具体地，

（1）index（表面是做index，实际是实时kmeans聚类、归类）

通过双塔，对所有item做实时的kmeans。实时计算所有item的cluster centroids、实时计算具体的item的embed，实时将item归结到最近的centroids上，并通过通用的、易得的、不依赖平台的梯度下降算法更新item embed和centroid embed。与此同时，会记录每个item所属哪个centroid。serving的时候，就能retrieve某个centroid下的所有item。

（2）DNN做排序

在训练的时候，这个是与前者并行的。serving的时候，（1）会召回最近cluster里的item，数量已经很少。在这里就可以大胆的用DNN做ranking。

如果您看过paper——Enhancing User Interest based on Stream Clustering and MemoryNetworks in Large-Scale Recommender Systems（https://arxiv.org/abs/2405.13238），就会对该paper有更深入的认识。他们可能是英雄所见略同。这个paper的idea比较简单易懂，应该对冷启item有较大的提升作用，值得尝试。

可信度：生产

推荐等级：推荐看看

请帮忙点赞、转发，谢谢。

个人观点，仅供参考
