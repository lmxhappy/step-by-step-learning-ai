# 推荐算法(2025.4.8)：阿里Masked Softmax Loss，生成式推荐别乱惩罚虚构物品

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484424&idx=1&sn=103d2fab460bb03ccfd332c5f1c2f0a3&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-08
> 本文拆自《推荐算法(2025.4.8)：魔鬼藏在细节里》，同期 3 篇：[Meta 片段信息 CTR 预估](../feature-cross/2025-04-08-meta-segment-ctr.md)、[阿里 Masked Softmax Loss](../generative-recommendation/2025-04-08-ali-masked-softmax-loss.md)、[阿里重排一致性原则](../multi-task-learning/2025-04-08-ali-rerank-consistency.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-04-08-ali-masked-softmax-loss.full.en.md)

当期前言：

（1）这篇的3篇paper都值得看，虽然paper中用到的技术可能不是很新颖，但是在魔鬼藏在细节里。

（3）第二篇的亮点在于对“无效token”的处理，让生成式推荐的正负样本学习更有效。

**魔鬼藏在细节里，这句话值得我们在工作中不断提醒自己！**

Masked Softmax Loss: Mitigating the Impact of Fictitious Items in Generative Recommendation

网址：https://arxiv.org/pdf/2504.04517

公司：阿里

方向：生成式推荐

解读：

本文提出了一个新的loss——Masked Softmax Loss（MSL），通过在损失计算中屏蔽无效token（即虚构物品描述对应的token），避免对这些token的惩罚，从而更好地聚焦于正向与负向物品的区分。就是在softmax计算概率的时候，分母不考虑无效token。

虚构物品（fictitious items）指的是在推荐系统中并不实际存在的物品，即那些并非系统物品集合一部分的物品描述。这些虚构物品通常是由大型语言模型（LLM）在生成过程中"想象"或"创造"出来的，而不是用户真正交互过的真实物品。

将所有虚构物品描述视为负样本，即使这些虚构物品可能与正向物品语义相似（如"钢铁侠：AI叛乱"与"钢铁侠1"），从而对LLM引入误导性训练信号。

心得：

随便乱惩罚是会出乱子的，无效token不是我们所希望的输出，但是不能惩罚它，要绕开它，比如，通过其它后处理的方法避开。

可信度：离线

推荐等级：推荐看看。
