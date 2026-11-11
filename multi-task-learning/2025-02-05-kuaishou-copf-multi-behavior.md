# 推荐算法(2025.2.5)：快手从组合优化视角做多行为推荐

> 原文：https://mp.weixin.qq.com/s/iXXwsZ46YhkqLlZ-OMbvNg
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-05
> 本文拆自《推荐算法paper（2025.2.5）》，同期 3 篇：[快手 COPF 多行为推荐](../multi-task-learning/2025-02-05-kuaishou-copf-multi-behavior.md)、[何向南团队 LLM 推荐深思偏好对齐](../llm4rec/2025-02-05-academic-llm-deliberative-preference.md)、[何向南团队 影响函数解决 CVR 延迟反馈](../loss-func/2025-02-05-academic-delayed-feedback-influence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-05-kuaishou-copf-multi-behavior.full.en.md)

Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation

网址：https://arxiv.org/pdf/2502.02232

公司：快手

方向：多行为推荐

解读：这篇论文聚焦多行为推荐问题，指出主流方法在行为融合和任务协调方面存在不足。为此提出基于组合优化视角的 COPF 框架 ，包含 COGCN 和 DFME 两部分。COGCN 通过对用户行为模式在不同阶段施加约束，解决多行为融合的组合优化问题；DFME 从特征和标签层面改进前向、反向传播，协调多行为预测任务间相关性。特别的，DFME，在多行为预测阶段，从特征和标签两个层面改进前向和后向传播过程。通过对比学习和行为拟合专家生成，减少特征分布偏差；在聚合时改进门控机制，避免标签分布差异带来的负迁移。

可信度：离线

推荐等级：值得看看
