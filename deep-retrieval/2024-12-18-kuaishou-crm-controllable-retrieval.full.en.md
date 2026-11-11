# Recommendation papers (2024.12.18): Kuaishou CRM: a controllable retrieval model that uses targets like watch time as conditions
> [中文版](2024-12-18-kuaishou-crm-controllable-retrieval.md)

> Source: WeChat post https://mp.weixin.qq.com/s/a1C-yVkShcnPD5DqaViEpw, published 2024-12-18.
> Split from the roundup "Recommendation papers (2024.12.18)"; same issue: [Kuaishou CRM controllable retrieval](../deep-retrieval/2024-12-18-kuaishou-crm-controllable-retrieval.full.en.md), [Meituan LLM rec semantic alignment](../llm4rec/2024-12-18-meituan-semantic-convergence.full.en.md).

CRM: Retrieval Model with Controllable Condition

Link: https://arxiv.org/pdf/2412.13844

Company: Kuaishou

Area: retrieval

Overview: Use information such as watch time as a condition for two-tower retrieval to improve retrieval quality. Compared with ranking models, retrieval is limited in multi-objective modeling: the number of objectives has to be kept under control, so some objectives that are not modeled are used as conditions to improve the other objectives. The simple approach is to make it one of the inputs of the two towers. The more complex approach uses the Decision Transformer, an innovative reinforcement learning algorithm that treats the reinforcement learning problem as a sequence modeling problem. Personally, I think you just have to read more papers; once there is a new technique, you can put it to use and improve the results of the work at hand. Go read the Decision Transformer paper now.

Credibility: production

Recommendation: worth reading
