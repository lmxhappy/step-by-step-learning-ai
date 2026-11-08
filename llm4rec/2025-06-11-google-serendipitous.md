# 推荐算法(2025.6.11)：Google的意外推荐

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484786&idx=1&sn=1170e694c1f7a2c6759cebf02e0f4945&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-06-11

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 Serendipitous Recommendation with Multimodal LLM

网址：https://arxiv.org/pdf/2506.08283

公司：google

方向：LLM+i2i

解读：

通过大模型预测既相关又新奇、令人惊喜的内容，激发用户探索并提升体验。即通过大模型，预测用户偶然性兴趣（cluster）的描述，缩小推荐范围。

具体的，

（1）使用层次树状聚类方法，将项目分为不同层级的兴趣簇，每个簇带有相应的文字描述。

（2）生成下一个兴趣簇的描述，并找到兴趣簇的ID

使用平台最受欢迎的视频，微调多模态大模型，利用微调后的MLLM理解用户兴趣和多模态内容，生成下一个偶然性兴趣簇（cluster）的描述，将其与层次树状聚类中的node（cluster）进行match，找到最match的node就是相应的cluster。

从用户日志中提取高满意度的偶然性（上下文视频，推荐视频）对，用于微调MLLM。MLLM通过处理视频帧、缩略图和文本（如标题、描述），实现对内容的深入理解。并提出链式思考（Chain-of-Thought，CoT）提示策略，通过结构化推理提升MLLM对视频内容的表示能力。

（3）低层推荐：将MLLM生成的簇ID输入传统推荐模型（如基于Transformer的序列模型），高效检索具体推荐项目。

在底层，簇ID通过以下方式限制基于Transformer的序列模型：

* 嵌入输入：簇ID被转换为嵌入向量，加入用户交互序列作为上下文或条件。

* 动作空间限制：Transformer的输出层仅考虑预定簇内的候选物品，过滤掉其他簇的项。

* 高效检索：通过预计算簇内物品的嵌入或索引，模型快速排序或选择最相关的推荐。

心得：大模型辅助推荐模型的一个方法。

可信度：生产

推荐等级：推荐看看

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.6.10)：Meta的离线召回新方法](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484780&idx=1&sn=04d946019642bef0239567c2a1af97e1&scene=21#wechat_redirect)

■ [推荐算法(2025.6.9)：Meta的基于多模态知识蒸馏的I2I方法](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484773&idx=1&sn=db2b0b912f649427e1e076a8293e301a&scene=21#wechat_redirect)

■ [推荐算法(2025.6.6)：阿里的基于Transformer的推荐系统优化](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484767&idx=1&sn=6176061ab42a6bd181f669732c24b975&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
