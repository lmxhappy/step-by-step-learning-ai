# 推荐算法(2025.1.24)：华为全栈优化LLM理解超长用户行为序列

> 原文：https://mp.weixin.qq.com/s/ftA6TyeyHnfAcipTxg1Q0Q
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-24
> 本文拆自《推荐算法paper（2025.1.24）》，同期 2 篇：[亚马逊 LLM 驱动的 RL 策略探索](../llm4rec/2025-01-24-amazon-llm-policy-exploration.md)、[华为 LLM 长序列理解（SUBR）](../llm4rec/2025-01-24-huawei-llm-lifelong-sequence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-01-24-huawei-llm-lifelong-sequence.full.en.md)

Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation

网址：https://arxiv.org/pdf/2501.13344

公司：华为

方向：长序列+LLM

解读：强化大语言模型更好的处理长序列。存在的问题是大语言模型无法从长用户行为序列的纯文本语境中有效提取有用信息，即便该语境长度远低于大语言模型的语境限制。本paper通过两方面：（1）SUBR：从长序列里筛选语义相关的序列item；（2）增强LLM模型。后者，通过将传统推荐模型的item ID信息，与item的文本信息，做对齐，从而让LLM具有了对ID信息更好的感知和处理能力。

可信度：离线

推荐等级：推荐看看。特别是SUBR部分。
