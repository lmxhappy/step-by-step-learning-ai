# 推荐算法(2025.2.5)：何向南团队把大模型 reasoning 用到推荐，提出深思式推荐

> 原文：https://mp.weixin.qq.com/s/iXXwsZ46YhkqLlZ-OMbvNg
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-05
> 本文拆自《推荐算法paper（2025.2.5）》，同期 3 篇：[快手 COPF 多行为推荐](../multi-task-learning/2025-02-05-kuaishou-copf-multi-behavior.md)、[何向南团队 LLM 推荐深思偏好对齐](../llm4rec/2025-02-05-academic-llm-deliberative-preference.md)、[何向南团队 影响函数解决 CVR 延迟反馈](../loss-func/2025-02-05-academic-delayed-feedback-influence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-05-academic-llm-deliberative-preference.full.en.md)

Large Language Models for Recommendation with Deliberative User Preference Alignment

网址：https://arxiv.org/pdf/2502.02061

公司：何向南。虽然带着meta的作者，但是不是一作和通讯作者。相比，何向南的名声更大。

方向：LLM推荐+reasoning

解读：就是借用了最近大模型的reasoning到推荐系统。对用户历史行为，通过LLM模型做reasoning，推理出用户的喜好。然后再进行预测。

提出了一个reasoning应用在推荐系统上的术语——Deliberative Recommendation task.这个值得我们学习，就是胆子要大，要大胆提出一些的idea，如研究的分支方向，万一成功了呢。比一味的follow更有意思。另外，不禁感慨，他们团队，新概念应用的手速真是快啊。当然，最重要的是，我觉得这个idea的应用是靠谱的。

可信度：离线

推荐等级：推荐看看
