# 推荐算法(2025.2.3)：Google协同扩散模型，用评论构造伪用户提升用户表征

> 原文：https://mp.weixin.qq.com/s/qvI9Ac5AJ96QwL3DoSPP5Q
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-02-03
> 本文拆自《推荐算法paper（2025.2.3）》，同期 2 篇：[Amazon CHARM 分层多字段检索](../deep-retrieval/2025-02-03-amazon-charm.md)、[Google 协同扩散推荐](../representation-quality/2025-02-03-google-collaborative-diffusion.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-02-03-google-collaborative-diffusion.full.en.md)

Collaborative Diffusion Model for Recommender System

网址：https://arxiv.org/pdf/2501.18997

公司：google

方向：diffusion推荐

解读：通过diffusion，引入评论数据，提升用户的表征。3个大的步骤：（1）伪用户生成：将物品评论中的每个特征（单词）视为一个伪用户，通过 TF - IDF 计算特征重要性并归一化，构建伪用户交互向量。这既能融入现有 CF 框架，又能借助 DR 去噪能力，实现物品侧信息与用户信号互补，提升推荐质量。（2）个性化邻居识别：基于行为相似性，为每个查询用户从真实用户和伪用户中分别识别出个性化的前个邻居，使用余弦距离确定邻居集合，且这些邻居集合在预处理时计算并缓存，便于模型训练和推理高效使用。（3）协同信号聚合：在扩散过程中，聚合真实和伪邻居的偏好信息（PI），得到更细粒度的用户偏好表示 ，通过平均池化、行为相似性、参数化建模三种方法计算邻居注意力分数，为模型提供了灵活选择。同时，据此重新制定目标函数优化模型。

可信度：离线

推荐等级：有时间看看
