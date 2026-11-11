# 推荐算法(2024.12.18)：快手CRM：把时长等目标作为条件的可控召回模型

> 原文：https://mp.weixin.qq.com/s/a1C-yVkShcnPD5DqaViEpw
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-18
> 本文拆自《推荐算法paper（2024.12.18）》，同期 2 篇：[快手 CRM 可控召回](../deep-retrieval/2024-12-18-kuaishou-crm-controllable-retrieval.md)、[美团 LLM 推荐语义对齐](../llm4rec/2024-12-18-meituan-semantic-convergence.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-18-kuaishou-crm-controllable-retrieval.full.en.md)

CRM: Retrieval Model with Controllable Condition

网址：https://arxiv.org/pdf/2412.13844

公司：快手

方向：召回

解读：将时长等信息作为双塔召回的条件，提升召回效果。与排序模型相比，召回在多目标建模上是受限的，要控制目标数量，将部分不建模的目标，作为条件提升其它目标。简单的做法就是作为双塔的一个输入。复杂的做法是利用Decision Transformer——一种创新的强化学习算法，它将强化学习问题视为序列建模问题。个人觉得，还得是多看paper，有了新技术就可以利用起来，提升手头工作的效果，还不赶紧看Decision Transformer paper去。

可信度：生产

推荐等级：值得看看
