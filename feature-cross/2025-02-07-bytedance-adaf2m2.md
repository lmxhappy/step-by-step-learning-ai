# 推荐算法(2025.1.28)：字节AdaF2M2，把特征交互和序列建模结合起来

> 原文：https://mp.weixin.qq.com/s/pQgDNxLJ7vBt_QUB4safyA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-07（标题日期 2025.1.28）
> 本文拆自《推荐算法paper（2025.1.28）》，同期 3 篇：[字节 Long-Term Interest Clock](../behavior-sequence-modeling/2025-02-07-bytedance-long-term-interest-clock.md)、[字节 AdaF2M2](../feature-cross/2025-02-07-bytedance-adaf2m2.md)、[腾讯 PatchRec](../llm4rec/2025-02-07-tencent-patchrec.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-07-bytedance-adaf2m2.full.en.md)

AdaF2M2: Towards Feature-adaptive and Multi-level Model Mixture for Sequential Recommendation

网址：https://arxiv.org/pdf/2501.15816

公司：字节

方向：特征

解读：字节的另一个创新点：将特征工程中的特征交互，与序列建模结合起来。具体来说，

（1）特征掩码机制：当模型处理用户行为序列时，随机掩盖部分特征维度，迫使模型学习更鲁棒的特征表示，从而提高模型在面对不完整或噪声数据时的泛化能力。这一机制特别适用于处理用户行为序列中的稀疏性和不确定性问题。

（2）状态感知适配器：根据当前用户状态（如兴趣变化、上下文信息等），动态调整特征融合策略。这一适配器能够捕捉用户状态的细微变化，并据此优化特征表示，从而提高推荐的准确性和个性化程度。如果没看错的话，看上去是pepnet的epnet。

可信度：生产

推荐等级：推荐看看
