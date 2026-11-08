# 推荐算法(2025.5.12)：Pinterest的小发现

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484641&idx=1&sn=f9eb875e09957224758daf49b3536b24&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-05-12

嘿，记得给“稳扎稳打学AI”添加星标

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

[本公众号paper推荐原则](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484097&idx=1&sn=18322c279290b555eb030e12953b7c7f&scene=21#wechat_redirect)

1 The Evolution of Embedding Table Optimization and Multi-Epoch Training in Pinterest Ads Conversion

网址：https://arxiv.org/pdf/2505.05605

公司：Pinterest

方向：嵌入表优化

解读：提出了一个优化推荐模型的embedding layer的方法，就是调整其学习率，使其不同于主体网络的。包括两个小方法，提高该layer的学习率，但是降低其低频特征值的学习率。两者是结合在一起的，前者是对所有特征，后者是对部分特征。换句话说，低频特征值是叠加两者的影响。具体的，

（1）提高该layer的学习率：针对嵌入表设置更高的层特定学习率，基于Adam优化器，保持其他密集参数的学习率不变。

（2）频率自适应学习率方法：针对嵌入表中不常更新的行（低频ID）降低学习率，以缓解多轮训练过拟合，同时保持高频ID的性能。学习率根据行的相对对数频率进行缩放（log scaling），优于线性缩放（linear scaling）。

为什么中低频行更新次数少更容易引起过拟合呢？

由于低频行基于少量样本更新，基于极少样本的梯度更新具有高方差（variance），其嵌入向量容易"记住"这些样本的特定模式或噪声，而非泛化的特征表示。例如，一个低频广告ID可能只出现在几个正样本中，模型可能过度优化其嵌入以拟合这些样本，导致对其未来样本的预测偏差。相反，训练频率多的行（高频行）过拟合的可能性较低，主要原因包括：

* 低方差更新：高频行基于大量样本更新，梯度估计稳定，嵌入向量趋向泛化表示。统计稳定性：高频行 的嵌入向量基于大量样本更新，符合大数定律，梯度估计更接近真实分布，减少过拟合风险。
* 多样上下文：高频行覆盖多种场景，增强泛化能力。
* 对稀疏标签的鲁棒性：高频行关联更多正样本，减少对少数样本的过拟合。
* 高学习率的安全性：高频行能承受高学习率（如 Sparse Optimizer），加速收敛而不易过拟合。

可信度：生产

推荐等级：有实践价值——客官可以试试

请帮忙点赞、转发，谢谢。您认为哪篇论文最有潜力应用到实际业务中？

## 欢迎干货投稿 \ 论文宣传 \ 合作交流

推荐您继续阅读：

■ [推荐算法(2025.5.8)：ebay、字节和小红书的精彩工作](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484636&idx=1&sn=8ab523471efbb4cd3e03edb68ca79dea&scene=21#wechat_redirect)

■ [推荐算法(2025.5.5)：Google的行为序列增强建模](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484629&idx=1&sn=46d01e06bb13402cb70a638b9f2f02fe&scene=21#wechat_redirect)

■ [推荐算法周报(2025.5.4)：工业风就是简单有效重细节](https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484623&idx=1&sn=a3eaaa163da1ba5f8485a837d996b0d7&scene=21#wechat_redirect)

历史文章索引：https://docs.qq.com/sheet/DYXZTZ0tseHRyZGtt?tab=BB08J2

由于公众号试行乱序推送，您可能不再准时收到稳扎稳打学AI的推送。为了第一时间收到本号的干货内容，请将本号设为星标，以及常点文末右下角的“在看”。

喜欢的话点个在看吧👇

个人观点，仅供参考
