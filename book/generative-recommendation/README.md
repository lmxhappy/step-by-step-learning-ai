# 深入理解生成式推荐：设计原则与上线判断

> 开源书 v0.1（草稿） · 微信公众号「稳扎稳打学AI」配套  
> 作者：刘明星（Mingxing Liu） · [LinkedIn](https://www.linkedin.com/in/mingxing-liu-b31656284) · 微信 `lmxhappy`  
> 写法对齐李博杰《深入理解 AI Agent》开源书路子：**一个公式 · 三根支柱 · 先发后改 · 可验证清单当「实验」**

## 一句话定位

这不是又一本经典搜广推教材（双塔 / 精排特征 / 传统 CTR）。

**生成式推荐是新范式。** 本书只回答：工业上怎么理解它、怎么读公司论文、怎么裁成能上线的一刀。

## 核心公式（对标 Agent = LLM + 上下文 + 工具）

**生成式推荐 = 语义标识（SID） + 生成器 + 条件注入 + 服务约束（Harness）**

| 层 | 直觉 | 工业里常落地成 |
| --- | --- | --- |
| 语义标识 | 「词表 / 商品怎么变成能生成的 token」 | RQ-VAE / 分层 SID / tokenizer |
| 生成器 | 「大脑怎么吐出下一项或整页」 | 自回归 / 扩散 / 重排生成 |
| 条件注入 | 「眼睛看到什么才生成」 | 行为序列、Reason、query、上下文 |
| 服务约束 | 「怎样才不会在线上炸」 | beam、KV、近线刷新、失败回退 |

> 实践在前，命名在后：先搞清这四块各自解决什么工程问题，再去追 TGR / CONGRATS / OneLA 等专有名词。

## About (English)

Open book on **industrial generative recommendation** (not classic CTR RecSys). One formula, three pillars, shipping judgment. Companion to WeChat「稳扎稳打学AI」.

```bash
git clone https://github.com/lmxhappy/step-by-step-learning-ai.git
cd step-by-step-learning-ai/book/generative-recommendation
```

## 李博杰路子 → 本书怎么抄

| 他的 Agent 书 | 你的生成式推荐书 |
| --- | --- |
| 一门课/一线实践 → 开源书 | 公众号工业解读 → 开源书 |
| 一个公式扛全书 | 上面四元公式 |
| 三根支柱 + 进阶方向 | 标识 / 生成 / 条件，进阶：重排、Serving、评估 |
| 100+ 可跑实验 | **每章「可验证清单」**（指标对照、消融、上线门禁） |
| whisper coding | 口述要点 → Agent 初稿 → 你审判断 |
| GitHub + PDF 先发后改 | 同左；星标当进度条 |

## 全书骨架：三根支柱 + 三个进阶

### 支柱一 · 语义标识（看见可生成的商品世界）
- 第 2 章 · SID / Tokenizer：为什么要标识化  
- 第 3 章 · 标识怎么学：冲突、长尾、唯一性（工业论文作案例）

### 支柱二 · 生成器（怎么吐出下一项）
- 第 4 章 · 生成目标与似然陷阱（CONGRATS 等）  
- 第 5 章 · 场景生成：直播 / 电商 Feed（OneLive、Sigma 等）

### 支柱三 · 条件注入（生成时看什么）
- 第 6 章 · 序列与上下文  
- 第 7 章 · 离线 Reason、线上只查表（TGR-Reason）

### 进阶一 · 重排与整页
- 第 8 章 · 生成式重排 / 整页生成（与「可替换的精排背景」如何切开）

### 进阶二 · Serving Harness
- 第 9 章 · 大 beam、线性注意力、KV / 近线刷新（OneLA 等）

### 进阶三 · 评估与自动化迭代
- 第 10 章 · 指标口径、假阳性、A/B 读法  
- 第 11 章 · Auto Research 与生成式推荐实验循环（可选，连 A-MLE）

### 总纲与收束
- 第 0 章 · [怎么读工业生成式推荐论文](./ch00-how-to-read.md)  
- 第 1 章 · 范式地图：从排序打分到生成（待写）  
- 第 12 章 · 评审一页纸 + 下一步读什么（待写）

## 章节状态与素材

| 章 | 标题 | 状态 | 素材 |
| --- | --- | --- | --- |
| 0 | 怎么读 | 草稿 | [ch00-how-to-read.md](./ch00-how-to-read.md) |
| 1 | 范式地图 | 待写 | 新写 |
| 2–3 | SID 支柱 | 待写 | sid/、PayPal SID、HiGR/BARGE 背景 |
| 4 | 似然陷阱 / 重排生成 | 待搬 | [kuaishou-congrats.md](../../generative-recommendation/kuaishou-congrats.md) |
| 5 | 场景生成 | 待搬 | [kuaishou-onelive.md](../../generative-recommendation/kuaishou-onelive.md)、Sigma |
| 6–7 | 条件 / Reason | 待搬 | [tencent-tgr.md](../../generative-recommendation/tencent-tgr.md) |
| 8 | 整页 / 重排进阶 | 待写 | CONGRATS 延伸 |
| 9 | Serving | 待写 | OneLA、华为 HBF |
| 10–11 | 评估 / Auto Research | 待写 | 方法论 + auto-research/ |
| 12 | 一页纸与书单 | 待写 | 模板 |

## 每章固定结构（对标他的「原则 + 实验」）

1. **工程问题**：没有这个东西时线上疼在哪  
2. **原则**：可迁移的判断（实践在前）  
3. **工业案例**：1 篇公司论文深读（增量一刀）  
4. **指标怎么读**  
5. **实验（清单）**：对照表 / 消融序 / 上线门禁（读者可自检，不强制训练代码）  
6. **思考题**

## 生产节奏（抄 whisper coding）

1. 你口述本章「工程问题 + 增量一刀 + 三个数」（5–10 分钟）  
2. Agent 按模板出初稿（可基于已有公众号 md）  
3. 你只改判断句与口径  
4. push；定期导出 PDF  

先发目录与第 0–1 章 → 再搬 TGR / CONGRATS / OneLive → 再补 SID 与 Serving。

## 和经典搜广推书的边界

| | 经典搜广推书 | 本书 |
| --- | --- | --- |
| 范式 | 打分 / 排序 | **生成** |
| 主角 | 特征与损失 | SID、生成器、条件、Serving |
| 成功标准 | 讲全配方 | **能立项、能量化、能回退** |

## 许可证

与仓库根目录 [LICENSE](../../LICENSE) 一致。
