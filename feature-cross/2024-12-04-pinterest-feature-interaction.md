# 推荐算法(2024.12.4)：Pinterest 工业约束下的特征交叉经验

> 原文：https://mp.weixin.qq.com/s/EZNw3eYCg2p3qbzZ-6Awuw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-04
> 本文拆自《推荐算法paper（2024.12.4）》，同期 2 篇：[Pinterest 特征交叉经验](../feature-cross/2024-12-04-pinterest-feature-interaction.md)、[沃尔玛线上线下行为序列](../behavior-sequence-modeling/2024-12-04-walmart-instore-online-sequence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-04-pinterest-feature-interaction.full.en.md)

Improving feature interactions at Pinterest under industry constraints

网址：https://arxiv.org/pdf/2412.01985

公司：Pinterest

方向：特征交叉

解读：介绍了pinterest特征交叉的经验。包括特征的处理和特征交叉。（1）特征处理：稠密特征归一化处理；稀疏特征，选择合适的embed size，并且对大size的其进行降维；稀疏特征和嵌入特征的embed都要做L2 norm。（2）特征交叉：DCNv2。还有其它聊，如面临的challenge，如何解的，这个是满满工业风的paper，特别值得推荐。

可信度：生产

推荐等级：强烈推荐。
