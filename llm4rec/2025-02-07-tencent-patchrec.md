# 推荐算法(2025.1.28)：腾讯PatchRec，用补丁压缩让大模型处理长行为序列

> 原文：https://mp.weixin.qq.com/s/pQgDNxLJ7vBt_QUB4safyA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-07（标题日期 2025.1.28）
> 本文拆自《推荐算法paper（2025.1.28）》，同期 3 篇：[字节 Long-Term Interest Clock](../behavior-sequence-modeling/2025-02-07-bytedance-long-term-interest-clock.md)、[字节 AdaF2M2](../feature-cross/2025-02-07-bytedance-adaf2m2.md)、[腾讯 PatchRec](../llm4rec/2025-02-07-tencent-patchrec.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-07-tencent-patchrec.full.en.md)

PatchRec: Patching and Compression for Efficient LLM-Based Sequential Recommendations

网址：https://arxiv.org/pdf/2501.15087

公司：腾讯

方向：大模型+长序列

解读：腾讯关于大模型+长序列的创新paper。主要是利用LLM的强大能力，解决长序列建模的问题。

通过多粒度压缩和两阶段训练，

（1）分层压缩：通过引入多粒度的补丁压缩机制，将原始行为序列中的每个项目转换为不同粒度的补丁，并通过层级压缩来减少输入token数量。同时，patch之间可以进一步分层压缩，以实现更高效的序列表示。 （2）两阶段训练：包括补丁预训练和补丁微调。补丁预训练阶段通过同时学习原始和压缩数据样本，让模型建立项目补丁与原始文本标记的对应关系；补丁微调阶段提出两种不同压缩粒度的策略 PFT-I 和 PFT-S，使模型适应多粒度和时间感知的压缩格式，捕捉用户兴趣的时间衰减。PFT-S，使模型适应多粒度和时间感知的压缩格式，捕捉用户兴趣的时间衰减。

可信度：离线

推荐等级：推荐看看
