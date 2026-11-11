# 推荐算法(2025.4.1)：百度大小模型协作排序CoRanking

> 原文：https://mp.weixin.qq.com/s?__biz=MzkzNDYyNjM1NA==&mid=2247484391&idx=1&sn=ef838380260b349648ae9d9bc69b52ab&scene=21#wechat_redirect
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-04-01（修改于 2025-04-01）
> 本文拆自《推荐算法(2025.4.1)：工业风拉满！Meta、百度、微软的成果》，同期 3 篇：[Meta 位置偏置物品选择](../deep-retrieval/2025-04-01-meta-position-biased-retrieval.md)、[微软 Wasserstein 重排](../loss-func/2025-04-01-microsoft-wasserstein-rerank.md)、[百度 CoRanking 大小模型协作排序](../llm4rec/2025-04-01-baidu-coranking.md)。

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2025-04-01-baidu-coranking.full.en.md)

CoRanking: Collaborative Ranking with Small and Large Ranking Agents

网址：https://arxiv.org/pdf/2503.23427

公司：百度

方向：rerank

解读：

其核心 idea 是提出了一种名为 CoRanking 的协作式排序框架，具体如下：

（1）全局用小模型。先使用小型重排序器对所有候选段落进行预排序，将相关段落到列表靠前位置。

（2）局部用大模型。再利用大语言模型列表式重排序器对小型重排序器输出的排名靠前的段落进行重排序，以此提高整体排序效率。

（3）纠偏。针对大语言模型列表式重排序器存在的位置偏差问题，引入通过强化学习训练的段落顺序调整器，对小型重排序器输出的段落重新排序，使其符合大语言模型对段落顺序的偏好，最终实现高效且有效的搜索结果排序。

心得：华为的 paper 说大模型赋能推荐算法是有用的，但是提高效率是落地的重点。这个 paper 是落地的一个体现。满满的工业风。

可信度：离线

推荐等级：有时间看看。没时间，看看解读和心得就够了。
