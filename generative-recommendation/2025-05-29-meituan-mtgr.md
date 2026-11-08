# 推荐算法(2025.5.29)：美团的生成式推荐落地方案

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484715&idx=1&sn=8471058be40bd73c024774e2d09d3ba2&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-05-29（修改于 2025-05-29）

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

👆 [本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 MTGR: Industrial-Scale Generative Recommendation Framework in Meituan

网址：https://www.arxiv.org/pdf/2505.18654v2

公司：美团

方向：生成式推荐

解读：

本文介绍了HSTU在美团的落地工作，在模型方面是微创新，主要创新是在推理阶段，而这也是为了落地而做。推理阶段就是深度使用Nvidia的优良feature，挖掘和发挥其GPU的推理能力。

微创新有：

（1）保留交叉特征：将用户特征、历史行为序列、实时交互和候选者特征（包括交叉特征）转化为统一令牌序列，交叉特征被整合进候选者令牌中。

（2）组层归一化：按领域分组对不同领域的token进行归一化，确保每个领域内的token分布相似，通常调整为均值0、方差1的分布，从而在自注意力计算前对齐不同领域的语义空间。

（3）动态掩码策略：MTGR模型用来处理令牌序列的一种方法，主要目的是避免信息泄露，同时提升模型性能。它的核心思想是根据令牌的类型和时间关系，灵活控制哪些令牌可以“看到”其他令牌的信息。

推理阶段的创新有：

（1）通过集成Nvidia提供的深度优化的Cutlass-based HSTU kernel，支持变长序列的输入无需padding，大幅提升了Attention的计算效率，单算子性能相较于Triton版本提升2~3倍。

（2）引入动态BS，每张卡的BS根据实际数据的序列长度动态调整，保证计算量（total_tokens）基本相同。因为少数用户的序列很长，大部分用户的序列都比较短，每张卡拿到的用户数相同，但由于序列长度不同实际的计算量差别较大。而每个step都要等负载最重的卡计算完，所有卡才能进行梯度同步。

（3）选择TensorRT作为模型推理框架：TensorRT是Nvidia推出的推理优化框架，在业界广泛应用，具有较强的算子融合、低精度量化能力。

贴一下ab效果： 转换量提升 1.22%，点击率（CTR）提升 1.31%。同时，训练成本保持不变，推理成本降低 12%。

参考： https://static.rainfocus.com/nvidia/gtcs25/sess/1734091150868001dcjp/FinalPresPDF/S74073_1741683377945001YCts.pdf

心得： （1）这是生成式落地的一种方法，就是国内大厂将精力主要放在推理阶段，推理阶段的优化就是挖掘GPU及其API的能力，这个有Deepseek的成功可以借鉴。将模型的创新工作交给国外大厂，国内大厂做好follow和落地工作。

可信度：生产

推荐等级：推荐看看

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.5.28)：美团的一体推荐](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484708&idx=1&sn=2383a8dd82e93d20404d1f817a4d788a&scene=21#wechat_redirect)

■ [推荐算法(2025.5.27)：ebay的关键词召回很实用](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484702&idx=1&sn=d5c348c24be320abedabd30da864f1ed&scene=21#wechat_redirect)

■ [推荐算法(2025.5.26)：腾讯、美团的AB效果显著](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484696&idx=1&sn=52a577d039ccc497941425043935fc25&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容， 请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考

修改于2025年5月29日
