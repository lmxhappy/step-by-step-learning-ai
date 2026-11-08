# 推荐算法paper

> 原文：https://mp.weixin.qq.com/s/KFK4lZ5ygqU2tv10TU_6OA
> 作者：算法工程师刘明星（稳扎稳打学AI）| 发表于 2025-01-22

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

* 推荐标准：大都是大厂的、少量学术的ranking model。
* 更新频率：日更。
* 来源：arxiv.org。
* 阅读建议：优先看前面的。

```
1 推荐等级：现在只有两个——推荐看看，有时间看看。偶尔会"强烈推荐看看"。
2 * 推荐看看，做ranking的同学需要及时了解一下，对于工作有帮助。
3 * 有时间看看，如名。
4
5 可信度：包括离线和生产。
6 * 生产：能用到生产中的方法，有用性会相对较大，他人更原因用在自己的业务上。
7 * 离线：相比，离线会小一些。
```

1 Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation

网址：https://arxiv.org/pdf/2501.11671

公司：百度

方向：跨域推荐+Diffusion+用户冷启动

解读：主要研究跨域推荐（CDR）中利用偏好引导扩散模型解决冷启动问题。就是对target域的user的embedding使用Diffusion的前向加噪和后向的去噪，提高user embedding的质量。在后向去噪的过程中，使用source域的交互历史为引导信号，即作为input。可见，Diffusion用来提高embed质量，解决用户冷启动，越来越普遍了。比如最近的DiffGraph: Heterogeneous Graph Diffusion Model（网址：https://arxiv.org/pdf/2501.02313），利用diffusion的强大祛噪能力，帮助GNN建模过程中的去噪。

可信度：离线

推荐：有空看看。

请帮忙点赞、转发，谢谢。

个人观点，仅供参考
