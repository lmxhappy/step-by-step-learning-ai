# 推荐算法(2024.12.24)：华为用LLM合成困难负样本提升稠密召回

> 原文：https://mp.weixin.qq.com/s/8UzNMNhhiqc8HL3SYfQ3GQ
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-24
> 本文拆自《推荐算法paper（2024.12.24）》，同期 4 篇：[华为多场景推荐基准 Scenario-Wise Rec](../cross-domain-recommendation/2024-12-24-huawei-scenario-wise-rec-benchmark.md)、[微软 RSLLM](../llm4rec/2024-12-24-microsoft-rsllm.md)、[Snap GraphHash 压缩 Embedding 表](../ml-infra/2024-12-24-snap-graphhash.md)、[华为 SyNeg LLM 合成困难负样本](../deep-retrieval/2024-12-24-huawei-syneg-llm-hard-negatives.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-24-huawei-syneg-llm-hard-negatives.full.en.md)

SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval

网址：https://arxiv.org/pdf/2412.17250

公司：华为

方向：召回+负样本生成+LLM

解读：本研究提出了一个框架，利用LLMs合成高质量的困难负样本。设计了一种多属性自我反思提示策略，引导LLMs生成困难负样本。

可信度：离线

推荐等级：有时间看看
