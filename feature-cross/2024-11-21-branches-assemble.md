# 推荐算法paper

> 原文：https://mp.weixin.qq.com/s/BlwnVYozm7rK0mxJkuP-8w
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2024-11-21

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

* 推荐标准：大都是大厂的、少量学术的。
* 更新频率：日更。
* 来源：arxiv.org。
* 阅读建议：优先看前面的。

1 Branches, Assemble! Multi-Branch Cooperation Network for Large-Scale Click-Through Rate Prediction at Taobao

网址：https://arxiv.org/pdf/2411.13057

公司：阿里淘宝

方向：特征交叉

解读：branch的意思是不同的特征交叉建模做法，比如dense隐性建模、cross线性建模，每一种建模方法就是一个branch。提出了一种多分支协作机制，就是branch之间互相协作。就像多组expert建模一样，特征交叉建模也在尝试增加并行的、互相协作的、互相分工的模块化方案。

可信度：生产。0.09point increase in CTR, 1.49% growth in deals, and 1.62% rise in GMV.

推荐等级：值得看看

个人观点，仅供参考
