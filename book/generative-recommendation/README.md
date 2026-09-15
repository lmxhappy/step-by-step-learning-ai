# 工业生成式推荐：新范式下的上线判断

> 开源笔记书 v0.1（草稿） · 微信公众号「稳扎稳打学AI」配套  
> 作者：刘明星（Mingxing Liu） · [LinkedIn](https://www.linkedin.com/in/mingxing-liu-b31656284) · 微信 `lmxhappy`

## 一句话定位

这不是又一本经典搜广推教材（双塔 / 精排特征 / 传统 CTR）。

**本书只写一件事：生成式推荐（Generative Recommendation）作为新范式，工业上怎么读论文、怎么对指标、怎么裁成能上线的一刀。**

核心公式：

**工业生成式推荐落地 = 选对论文 × 读准指标 × 裁成可上线的一刀**

## About (English)

An open notes-book on **industrial generative recommendation** — not another classic RecSys textbook. Focus: how to read company papers, interpret online metrics, and decide what to ship. Companion to WeChat「稳扎稳打学AI」.

```bash
git clone https://github.com/lmxhappy/step-by-step-learning-ai.git
cd step-by-step-learning-ai/book/generative-recommendation
```

## 和经典搜广推书的区别

| | 经典搜广推书 | 本书 |
| --- | --- | --- |
| 对象 | 召回 / 粗精排 / 特征交叉 | SID 生成、整页/序列生成、Reason 条件、生成式重排 |
| 叙事 | 模型配方大全 | **新范式下的判断力** |
| 交付 | 原理 + 有时带代码 | 笔记 + 指标口径 + 上线清单 |
| 更新 | 出版周期长 | GitHub 持续迭代 |

## 目录（v0.1）

| 章 | 标题 | 状态 | 素材 |
| --- | --- | --- | --- |
| 0 | [怎么读工业生成式推荐论文](./ch00-how-to-read.md) | 草稿 | 方法论 |
| 1 | 范式地图：从排序到生成 | 待写 | 总览 |
| 2 | 场景落地：直播（OneLive） | 待搬 | [kuaishou-onelive.md](../../generative-recommendation/kuaishou-onelive.md) |
| 3 | 离线 Reason、线上查表（TGR-Reason） | 待搬 | [tencent-tgr.md](../../generative-recommendation/tencent-tgr.md) |
| 4 | 生成式重排与似然陷阱（CONGRATS） | 待搬 | [kuaishou-congrats.md](../../generative-recommendation/kuaishou-congrats.md) |
| 5 | 搜索 / Query 生成 | 待写 | EAGER 等 |
| 6 | Serving：beam、线性注意力、KV | 待写 | OneLA 等 |
| 7 | SID：何时上、上哪种 | 待写 | 跨文对比 |
| 8 | 评审一页纸模板 | 待写 | 可公开简化版 |
| 9 | 假阳性与评估坑 | 待写 | reward hacking、新用户≠新 item |
| 10 | 下一步读什么 | 待写 | 与 arXiv 日报联动 |

## 每章固定四块

1. **场景**：解决什么业务问题  
2. **增量一刀**：真正新的是什么（其余当可替换背景）  
3. **指标怎么读**：口径、分母、线上数字  
4. **能不能上**：延迟、刷新、失败回退 |

## 生产约定

- 公开 paper + 工业判断；不写东家未公开系统  
- 以笔记为主，不以训练代码复现为主  
- 先发后改：v0.1 可缺章，目录先立住  

## 许可证

与仓库根目录 [LICENSE](../../LICENSE) 一致。
