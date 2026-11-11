# 推荐算法(2025.2.5)：何向南团队用影响函数解决 CVR 预估的延迟反馈

> 原文：https://mp.weixin.qq.com/s/iXXwsZ46YhkqLlZ-OMbvNg
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-05
> 本文拆自《推荐算法paper（2025.2.5）》，同期 3 篇：[快手 COPF 多行为推荐](../multi-task-learning/2025-02-05-kuaishou-copf-multi-behavior.md)、[何向南团队 LLM 推荐深思偏好对齐](../llm4rec/2025-02-05-academic-llm-deliberative-preference.md)、[何向南团队 影响函数解决 CVR 延迟反馈](../loss-func/2025-02-05-academic-delayed-feedback-influence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-05-academic-delayed-feedback-influence.full.en.md)

Addressing Delayed Feedback in Conversion Rate Prediction via Influence Functions

网址：https://arxiv.org/pdf/2502.01669

公司：何向南。

方向：广告cvr预测

解读：本文聚焦于转化率（CVR）预测中的延迟反馈问题 ，提出基于影响函数的延迟反馈建模框架（IF - DFM）。其核心思路是借助影响函数估计新数据和延迟转化数据对模型参数的影响，直接修改参数，避免重新训练。同时，通过将逆海森向量积计算转化为有限和二次规划问题，利用随机优化算法提升计算效率。一天同时发表两篇paper，我猜何老师一次性同意了两篇paper的发表。

可信度：离线

推荐等级：有时间看看
