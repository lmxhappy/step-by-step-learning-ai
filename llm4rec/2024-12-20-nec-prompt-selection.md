# 推荐算法paper

> 原文：https://mp.weixin.qq.com/s/THRqyDod0hndVt6jOLWxfA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-12-20

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

> [English version](2024-12-20-nec-prompt-selection.full.en.md)

* 推荐标准：大都是大厂的、少量学术的ranking model。
* 更新频率：日更。
* 来源：arxiv.org。
* 阅读建议：优先看前面的。

1 Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems

网址：https://arxiv.org/pdf/2412.14454

公司：NEC

方向：LLM推荐

解读：

这篇论文研究了如何为基于大型语言模型的推荐系统（LLM-RSs）选择合适的提示（prompts）。

（1）提示重要性： 在 LLM-RSs 中，提示的选择对推荐的准确性至关重要。

（2）没有万能提示： 没有哪个单一的提示在所有情况下都表现最佳，提示的选择需要考虑数据集的特性。

（3）提示选择方法： 论文提出了一种能够以最小验证数据实现高准确性的提示选择方法。就是从一堆提示词的模版里，通过自动化尝试，选出针对不同dataset的最好的提示词。

（4）成本效益策略： 论文还提出了一种使用高性能、低成本 LLM 来降低提示探索成本的方法。

可信度：离线

推荐等级：有时间看看

请帮忙点赞、转发，谢谢。

个人观点，仅供参考
