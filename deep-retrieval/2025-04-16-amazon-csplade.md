# 推荐算法(2025.4.16)：Amazon CSPLADE，用因果大模型做学习型稀疏检索

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484494&idx=1&sn=4765dee4aa47534d3892f6f9b9fdb89c&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-16
> 本文拆自《推荐算法(2025.4.16)：Pinterest & Amazon》，同期 2 篇：[Pinterest PinRec 生成式检索](../generative-recommendation/2025-04-16-pinterest-pinrec.md)、[Amazon CSPLADE 稀疏检索](../deep-retrieval/2025-04-16-amazon-csplade.md)。

关注我，每天为你精选推荐系统的最新好论文。

> [English version](2025-04-16-amazon-csplade.full.en.md)

CSPLADE: Learned Sparse Retrieval with Causal Language Models

网址：https://arxiv.org/pdf/2504.10816

公司：Amazon

方向：稀疏检索（LSR）

解读：

论文解决了将LLM用于LSR的两个主要挑战：训练不稳定性与单向注意力机制导致的性能次优。

什么是稀疏检索（LSR）？

通过预训练语言模型（如BERT或LLM）学习生成词汇表大小的稀疏向量，每个维度表示词汇表中某个词的“重要性”或“影响”，重要性非零的词（超过阈值）用于构建倒排索引，实现高效检索，索引大小远小于密集检索。

方法：将预训练因果LLM（如LLAMA-3.1-8B）作为骨干模型，生成词汇大小的稀疏向量，适配倒排索引以实现高效检索。具体的训练流程：

（1）适配阶段：用无标注文本进行轻量级预训练，优化因果语言建模损失与ReLU适配损失的组合。

（2）对比训练阶段：使用InfoNCE损失进行对比学习，结合硬负例挖掘和FLOPs正则化以增强稀疏性。

心得：利用了大模型的强大理解能力，让match召回焕发新机。

可信度：离线

推荐等级：有时间看看
