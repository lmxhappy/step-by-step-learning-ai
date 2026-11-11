# 推荐算法(2025.4.15)：Meta用无监督数据增强和图对比学习改进序列推荐

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484486&idx=1&sn=6687984fa2dd4d718ea0c55554f2747d&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-15（标题已修改；修改于 2025-04-15）
> 本文拆自《推荐算法(2025.4.15)：3篇大厂非大模型相关的论文》，同期 3 篇：[Meta URM 统一推荐模型](../deep-retrieval/2025-04-15-meta-urm-unified-model.md)、[Meta FROG 多模态好友推荐](../graph-recommendation/2025-04-15-meta-frog-friend-rec.md)、[Meta 无监督数据增强序列推荐](../behavior-sequence-modeling/2025-04-15-meta-unsupervised-seq-enhancement.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-04-15-meta-unsupervised-seq-enhancement.full.en.md)

Recommendation System in Advertising and Streaming Media: Unsupervised Data Enhancement Sequence Suggestions

网址：https://arxiv.org/pdf/2504.08740

公司：Meta

方向：序列建模

解读：

针对序列建模中的数据稀疏和噪声问题，通过图对比学习和无监督数据增强方法提升推荐性能。同时发明了一个兴趣提取融合方法。具体的，

（1）增强序列

使用生成对抗网络（GAN）生成逼真的用户-项目交互序列，丰富训练数据多样性。生成器从部分输入生成序列，判别器评估序列真实性，通过对抗训练优化。

加入多样性促进项，避免模式崩塌（mode collapse），确保生成数据质量。

（2）构建、优化项目关系图

从所有用户交互序列中建一个全局无向图，节点为项目，边权重反映项目在不同序列中的邻接关系。通过定义高阶邻接信息（如不同时间间隔的连接）并设置阈值过滤噪声，优化图结构。对热门项目减少边连接，以缓解流行度偏差。

（3）图对比学习

对关系图应用图对比学习，通过概率邻域采样生成子图，增强项目嵌入的鲁棒性。通过对比正样本（同一项目的增强表示）与负样本（其他项目），优化项目表示，捕捉全局关联。

（2）多兴趣提取与融合

使用改进的胶囊网络，结合目标注意力机制（Target-Attention），捕捉用户动态和多样化的兴趣。通过Transformer编码用户交互序列，结合动态路由更新兴趣胶囊，生成个性化用户偏好向量。

最终推荐分数基于用户偏好向量与目标项目嵌入的交互计算。

心得：GNN用于行为序列是一个好的方法，可以增加序列item的知识量，在多个paper里见到效果了。

可信度：离线

推荐等级：推荐看看
