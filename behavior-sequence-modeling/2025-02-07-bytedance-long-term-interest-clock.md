# 推荐算法(2025.1.28)：字节长期兴趣时钟，长序列建模加上细粒度时间感知

> 原文：https://mp.weixin.qq.com/s/pQgDNxLJ7vBt_QUB4safyA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-07（标题日期 2025.1.28）
> 本文拆自《推荐算法paper（2025.1.28）》，同期 3 篇：[字节 Long-Term Interest Clock](../behavior-sequence-modeling/2025-02-07-bytedance-long-term-interest-clock.md)、[字节 AdaF2M2](../feature-cross/2025-02-07-bytedance-adaf2m2.md)、[腾讯 PatchRec](../llm4rec/2025-02-07-tencent-patchrec.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-07-bytedance-long-term-interest-clock.full.en.md)

Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System

网址：https://arxiv.org/pdf/2501.15817

公司：字节

方向：长行为序列+时间建模

解读：在长序列建模的时候，实现细粒度的时间感知。借用了SIM模型的GSU和ESU的两步，在此基础上，都加上了对time的考虑。让我们回忆一下，GSU就是从长序列里search出来一个短子序列；ESU就是根据该短序列做兴趣建模。该paper的这两部分：（1）GSU：在SIM的GSU逻辑的同时，add上了时间相似度项；（2）ESU：在SIM的GSU逻辑的基础上，对序列item的embed做了concatenate，拼接了时间间隔的Δ, √Δ, Δ2, log(Δ + 1)），从而增加了时间对兴趣embed的影响。

这个paper是Interest clock: Time perception in real-time streaming recommendation system（https://arxiv.org/pdf/2404.19357）的后续paper。

用户的时间规律性是明显的，对此建模也越来越多了，见另外一个paper的解读——[推荐算法paper](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484070&idx=1&sn=9a53aa7c49f700d68e2c3a8011bf4aeb&scene=21#wechat_redirect)

这个paper属于微创新，短序列基础上借用SIM思想。虽说如此，也不错。

推荐等级：生产——抖音音乐

可信度：推荐看看
