# 推荐算法(2025.7.19)：Pinterest出品必是精品——GPT建模行为序列

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484902&idx=1&sn=2ebdcda5dceefb8d2608f48bdeaab839&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-07-19

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform

网址：https://arxiv.org/pdf/2507.12704

公司：Pinterest

方向：行为序列+LLM

解读：

本文训练了一个行为序列兴趣建模的模块，该模块单独预训练，后加入到主推荐模块微调。

（1）预训练

数据：收集两年用户活动数据（时间戳、动作类型、页面类型、项目 ID），序列长度上限为 16,000，切分为数百长度的片段。

模型：采用 GPT2 架构（带 Pre-LN），通过 infoNCE 损失函数进行下一标记预测，新增多标记预测（Multi Token Loss）和未来标记预测（Future-Token Loss），以捕捉用户兴趣的短期和长期变化。

优化：使用 ID 嵌入而非高维预训练嵌入，降低存储和计算成本；针对大词汇量（亿级项目 ID），采用对比学习损失。

（2）微调

将预训练模型集成到下游排名模型（如 DLRM 或 DCN），通过早期融合（将候选项目加入用户序列）增强用户-项目交互捕捉。微调中加入预训练损失和排名模型损失，学习率设置为排名模型的1/10，以保留预训练知识。冷启动优化：通过候选项目 ID 随机化和基于项目年龄的 dropout，改善对新项目的推荐性能。

（3）推理

DCAT 通过去重、KV cache 复用、固定序列长度、跳过冗余计算和 Triton 内核优化，显著降低了 PinFM 的推理耗时，使其从潜在的“很长”变为高效可控。

模型将用户行为序列的 self-attention计算部分的每层的K和V都缓存下来，后续只需要跟 candidate item每层的KV做cross attention，这样减少了推理时间。

另外，嵌入表托管在 CPU 集群，transformer 推理在 GPU 集群，推理路由高效传递数据，避免 GPU 内存瓶颈。

还有，嵌入表从 float16 量化为 int4，体积缩小至 31.25%，数据传输量减少，推理延迟降低 7%（因嵌入 I/O 负载减少）。

PinFM 被集成到主推荐模型（例如 Home Feed 和 Related Items Feed 的排序模型）后，一次查询的处理时间需满足数百毫秒的延迟。

贴一下AB：

（1）Home Feed（主页）：整体 Save 提升1.20%，页面 Save 提升2.60%，新鲜项目 Save 提升8.20%；

（2）Related Items Feed（相关项目推荐）：整体 Save 提升0.90%，页面 Save 提升2.10%，新鲜项目 Save 提升7.50%。

心得：

（1）将推荐系统中的用户行为序列建模为类似语言模型的序列预测任务，真是很不错的想法。特别是通过多标记预测和未来标记预测，可以捕捉用户兴趣的短期和长期变化。

（2）通过将预训练模型集成到下游排名模型中，可以显著提升推荐效果。特别是通过早期融合和冷启动优化，可以改善对新项目的推荐性能。

可信度：生产

推荐等级：有实践价值

<!-- 本文配图：页面图片元素仅返回占位图，图片链接未能取到 -->

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.7.16)：Meta的相关短视频推荐的相关性提升了12%](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484896&idx=1&sn=00e46baa7b3bedc78520a72f4360ee74&scene=21#wechat_redirect)

■ [推荐算法(2025.7.15)：淘宝的长序列召回新方法GMV增长4个点](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484890&idx=1&sn=a0f75d4b0b5396e9efb6ad35af31f804&scene=21#wechat_redirect)

■ [推荐算法(2025.7.10)：淘宝的意图建模纠偏方法AB效果两位数](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484884&idx=1&sn=8a0e3c23281188cab8a4660ece028c23&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
