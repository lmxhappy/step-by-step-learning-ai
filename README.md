# Step By Step Learning AI

[![GitHub stars](https://img.shields.io/github/stars/lmxhappy/step-by-step-learning-ai?style=social)](https://github.com/lmxhappy/step-by-step-learning-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mingxing%20Liu-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/mingxing-liu-b31656284)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=lmxhappy.step-by-step-learning-ai)](https://github.com/lmxhappy/step-by-step-learning-ai)
[![Hits](https://hits.sh/github.com/lmxhappy/step-by-step-learning-ai.svg)](https://hits.sh/github.com/lmxhappy/step-by-step-learning-ai/)

如果这个仓库对你有帮助，欢迎点击右上角 **Star ⭐** 支持一下，让更多人发现它！

## About (English)

Companion notes for the WeChat account **稳扎稳打学AI** (*Step by Step Learning AI*): structured write-ups of industrial papers in **recommendation systems, computational advertising, and search** (搜广推).

- **What this repo is:** paper interpretation notes synced with the WeChat posts (methods, takeaways, production signals) — organized by topic folders below.
- **What it is not:** a code-first or full experiment-reproduction warehouse. Occasional scripts/links appear inside a note when relevant.
- **Author:** Mingxing Liu · [LinkedIn](https://www.linkedin.com/in/mingxing-liu-b31656284) · WeChat: `lmxhappy` · [Zhihu column](https://zhuanlan.zhihu.com/stupid-ai)

```bash
git clone https://github.com/lmxhappy/step-by-step-learning-ai.git
```

### Visitors

[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=lmxhappy.step-by-step-learning-ai)](https://github.com/lmxhappy/step-by-step-learning-ai)
![Hits](https://hits.sh/github.com/lmxhappy/step-by-step-learning-ai.svg)

欢迎来到本仓库！这是微信公众号「稳扎稳打学AI」的配套资源库，聚焦推荐 / 广告 / 搜索论文解读，陪你一起保持进步、不断精进。

## 关于公众号“稳扎稳打学AI”
- **公众号介绍**：关注我，每天为你精选推荐算法paper，帮助你稳扎稳打地学习AI。公众号由刘明星（微信: lmxhappy）运营，内容聚焦于推荐、广告和搜索等。
![e](imgs/qcode.jpg)
- **paper推荐原则**：我们优先选择高影响力、实用性强的论文，包括但不限于LLM在推荐中的应用、搜索词推荐等实际案例分析。
- **知乎专栏**：更多内容可查看[稳扎稳打学AI - 知乎专栏](https://zhuanlan.zhihu.com/stupid-ai)，已更新200+篇内容，涵盖热门论文解读。


## 仓库内容

### 📖 开源小书（进行中）
- [**工业生成式推荐：新范式下的上线判断**](./book/generative-recommendation/)（v0.1 草稿）— 另起炉灶，不写经典搜广推大全，只写生成式推荐的工业阅读与上线判断。

这个仓库是公众号「稳扎稳打学AI」的**配套解读笔记库**（与公众号同步更新），主要包括：
- **论文解读 / 笔记**：针对搜广推相关论文的结构化解读（方法、心得、可信度等），按主题目录组织。
- **配图与索引**：文中插图与 README 目录导航，方便连读与检索。
- **说明**：当前以笔记为主，**不以可运行代码 / 实验复现为主**；若个别文章附带脚本或外链，会在文内单独说明。

## 如何开始
1. **关注公众号**：搜索“稳扎稳打学AI”或扫描二维码，获取每日论文推荐。
2. **克隆仓库**：`git clone https://github.com/lmxhappy/step-by-step-learning-ai.git`
3. **贡献**：欢迎提交 PR，分享你的论文笔记或纠错。我们一起进步！

## 目录结构 & 内容说明

### 📁 子目录导航

- **[跨域](./cross-domain-recommendation/)**
  跨领域推荐。
  - [MTFM — 异构多场景推荐 Foundation Model (美团)](./cross-domain-recommendation/meituan-mtfm.md)
  - [YouTube Music 跨域蒸馏 — 零样本跨域知识蒸馏，新歌收听+11% (YouTube)](./cross-domain-recommendation/youtube-music-cross-domain.md)

- **[用户行为序列建模](./behavior-sequence-modeling/)**
  用户的历史行为序列建模，包括超长序列建模和普通序列建模。
  - [STCA — 超长序列建模，完播率+3% (字节)](./behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/bytedance-stca.md)
  - [Ultra-HSTU — HSTU 2.0 超长序列建模 (Meta)](./behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/meta-ultra-HSTU.md)
  - [MoS — 序列聚类+MoE 搞定长序列 (Meta)](./behavior-sequence-modeling/meta-mos.md)
  - [Sample Is Feature — 序列样本分词，CTR+2% (美团)](./behavior-sequence-modeling/meituan-sample-is-feature.md)
  - [GenLI — 生成式长兴趣建模，RPM+1.6% (美团)](./behavior-sequence-modeling/meituan-genli.md)
  - [UxSID — 用聚类搞定超长序列建模，收入+0.3% (快手)](./behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/kuaishou-uxsid.md)
  - [TokenMinds — 用户生成式SID特征双输出表征 (YouTube)](./behavior-sequence-modeling/tokenminds.md)
  - [UniSGR — 增加用户SID特征，GMV+6% (阿里)](./behavior-sequence-modeling/alibaba-unisgr.md)
  - [CMSL — 用户行为序列转成纯净意图流 (Meta)](./behavior-sequence-modeling/meta-cmsl.md)
  - [EST — 异构特征统一建模，RPM+3% (阿里)](./behavior-sequence-modeling/alibaba-est.md)

- **[生成式推荐](./generative-recommendation/)**
  生成式推荐。
  - [CONGRATS — 生成式重排，图结构解码器+一致性训练，Long Views+2.18% (快手)](./generative-recommendation/kuaishou-congrats.md)
  - [TGR-Reason — 离线reason token注入，新用户曝光转化+13.09% (腾讯)](./generative-recommendation/tencent-tgr.md)
  - [OneLive — 生成式推荐落地直播场景 (快手)](./generative-recommendation/kuaishou-onelive.md)
  - [Sigma — 生成式推荐，GMV+8% (阿里)](./generative-recommendation/ali-sigma.md)
  - [VectorizingTrie — 生成式召回解码提速最高1000倍 (Google)](./generative-recommendation/google-static.md)
  - [GEM-Rec — 统一推荐与广告的生成式推荐 (Google)](./generative-recommendation/google-gem-rec.md)
  - [RCLRec — 稀疏目标的生成式推荐，广告收入+2% (阿里国际)](./generative-recommendation/ali-rclrec.md)
  - [CQ-SID — 搜索LLM生成式召回，贡献七成成交 (阿里)](./generative-recommendation/ali-cq-sid.md)
  - [DGI — SID与GR联合训练，RPM+1.11% (阿里)](./generative-recommendation/ali-dgi.md)

- **[SID](./sid/)**
  语义ID学习与生成式检索。
  - [QuaSID — SID 量化新方法，GMV+2% (快手)](./sid/kuaishou-quasid.md)
  - [AKT-Rec — 聚类相关特征提升长尾，GMV+3% (阿里)](./sid/alibaba-akt-rec.md)

- **[LLM4Rec](./llm4rec/)**
  LLM 在推荐/召回中的各类应用范式。
  - [LLM语义召回 — LLM as annotator (Meta)](./llm4rec/meta-llm-retrieval.md)
  - [LLM合成查询生成 — 数据增强 (Airbnb)](./llm4rec/airbnb-llm-synthetic-query.md)
  - [级联生成式LLM首页个性化 — LLM as ranker (Instacart)](./llm4rec/instacart-cascaded-generative.md)
  - [GenRec — LLM直接当精排，一次前向给全目录打分 (Netflix)](./llm4rec/netflix-genrec.md)

- **[特征交叉](./feature-cross/)**
  - [MGDIN — 特征交叉新方法，CTR+3% (阿里)](./feature-cross/ali-mgdin.md)
  - [SlimPer — 多层多槽位 Target Attention 增强 (Meta)](./feature-cross/meta-slimper.md)
  - [DANet — 折扣率显式建模，GMV+2% (阿里)](./feature-cross/ali-danet.md)
  - [CCFormer — 长序列层次化压缩+三字段定向交叉，广告收入最高+1.71% (腾讯)](./feature-cross/tencent-ccformer.md)

- **[特征选择](./feature-selection/)**
  特征重要性评估、低价值特征淘汰等特征治理类论文。
  - [LO-FAR — 免GPU筛特征，稀疏参数最多砍掉75% (Meta)](./feature-selection/meta-lo-far.md)

- **[I2I推荐](./item-to-item/)**
  I2I（Item-to-Item）推荐。
  - [DAIAN — 详情页 I2I 推荐，成交额+2% (阿里)](./item-to-item/alibaba-daian.md)

- **[双塔召回](./deep-retrieval/)**
  两塔架构的深度召回。
  - [CS3 — 双塔召回，广告收入最高+8% (快手)](./deep-retrieval/kuaishou-cs3.md)
  - [HILL — 树状聚类索引+深度召回，业务指标+2.57% (Meta)](./deep-retrieval/meta-hill.md)
  - [HSNN — 深度召回，业务指标+3% (Meta)](./deep-retrieval/meta-hsnn.md)
  - [RankGraph — 聚类用于图召回，降本增效 (Meta)](./deep-retrieval/graph-retrieval/rankgraph-2.md)

- **[损失函数](./loss-func/)**
  排序、分类等核心损失函数的创新与优化。
  - [VarBPR — 改进 BPR 成对损失，应对隐式反馈噪声](./loss-func/varbpr.md)

- **[表征学习](./representation-learning/)**
  嵌入表征质量、表示坍缩等问题的建模与优化。
  - [RankUp — 解决大规模排序模型表示坍缩 (腾讯)](./representation-learning/tencent-rankup.md)

- **[ML Infra](./ml-infra/)**
  机器学习基础设施 / 特征运维 / 模型部署等工程系统类论文。
  - [IEFF — 特征优雅下线，又快又稳 (Meta)](./ml-infra/meta-ieff.md)
  - [Versioned Late Materialization — 推荐工程架构/特征物化 (Meta)](./ml-infra/meta_late_materialization.md)

- **[Auto Research](./auto-research/)**
  用AI自动化算法研发循环本身：提优化方向、写代码、跑实验、闭环反馈。
  - [Astar — 模型进化机器人，自己提优化方向，GMV+4.86% (阿里)](./auto-research/ali-astar.md)
  - [A-MLE — 排序模型调优交给Agent，迭代吞吐翻几倍 (Meta)](./auto-research/meta-amle.md)

- **[长尾问题](./long-tail/)**
  长尾样本、稀疏特征、低频物品、新用户、新item的建模与优化。
  - [UTTSI — 特征随机丢弃+集成解决稀疏样本预测不准 (阿里)](./long-tail/uttsi.md)

## 联系方式
- 作者：刘明星（Mingxing Liu）
- LinkedIn：[mingxing-liu-b31656284](https://www.linkedin.com/in/mingxing-liu-b31656284)
- 微信：lmxhappy
- 知乎：[@思达-刘明星](https://www.zhihu.com/people/wang-wang-20-73)
- GitHub：[`lmxhappy/step-by-step-learning-ai`](https://github.com/lmxhappy/step-by-step-learning-ai)
- 反馈：欢迎在公众号留言或 issue 中讨论。

感谢你的关注！让我们稳扎稳打，学好AI！🚀