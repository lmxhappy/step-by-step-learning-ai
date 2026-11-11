# 推荐算法(2025.1.31)：阿里DRP，解耦相关性与偏好，重构电商搜索行为建模空间

> 原文：https://mp.weixin.qq.com/s/k7nEXZJqXf6nJ96EkpgPzw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-04（标题日期 2025.1.31）
> 本文拆自《推荐算法paper（2025.1.31）》，同期 3 篇：[阿里 DRP 搜索行为建模空间重构](../behavior-sequence-modeling/2025-02-04-ali-drp-search-behavior.md)、[Meta HyperZero 融合公式自动调参](../multi-task-learning/2025-02-04-meta-hyperzero.md)、[Amazon 序列推荐最小最大群体公平](../loss-func/2025-02-04-amazon-minimax-group-fairness.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-04-ali-drp-search-behavior.full.en.md)

Behavior Modeling Space Reconstruction for E-Commerce Search

网址：https://arxiv.org/pdf/2501.18216

公司：阿里

方向：搜索

解读：提出DRP 框架，通过两个核心组件来重构行为建模空间。首先是偏好编辑，通过寻找合适的正交低秩矩阵，从偏好预测中去除相关性的影响，得到更纯净的用户偏好，重构出细粒度的建模空间。其次是双级自适应融合，包括全局自适应融合和局部自适应融合。全局自适应融合通过引入可学习参数，区分不同相关性 - 偏好状态的模式；局部自适应融合进一步拟合相似状态下不同行为信号的差异，使模型能更细致地捕捉用户行为模式。

解藕是最近多年来work的方法，之所以work是因为能够更加均衡多个因素的影响。

可信度：在线

推荐等级：推荐看看
