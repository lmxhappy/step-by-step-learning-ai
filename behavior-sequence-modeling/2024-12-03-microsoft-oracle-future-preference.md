# 推荐算法(2024.12.3)：微软用未来交互引导序列推荐的用户偏好建模

> 原文：https://mp.weixin.qq.com/s/Q7kLmvPiR8p55vzGOAoK5Q
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-03
> 本文拆自《推荐算法paper（2024.12.3）》，同期 2 篇：[华为大推荐模型扩展规律](../generative-recommendation/2024-12-03-huawei-large-rec-scaling-laws.md)、[微软未来信息引导序列推荐](../behavior-sequence-modeling/2024-12-03-microsoft-oracle-future-preference.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-03-microsoft-oracle-future-preference.full.en.md)

Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation

网址：https://arxiv.org/pdf/2412.00813

公司：微软

解读：现有的顺序推荐方法仅使用从当前互动之前的用户互动中提取的过去信息，本文将未来信息（就是该样本后面时间互动的item）当看作是一种后验信息，提取出未来信息，用以引导模型对过去信息进行训练。从而希望更全面的捕捉用户偏好的动态变化。

可信度：离线

推荐等级：推荐看看
