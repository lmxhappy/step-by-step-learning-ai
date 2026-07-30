# Step By Step Learning AI

[![GitHub stars](https://img.shields.io/github/stars/lmxhappy/step-by-step-learning-ai?style=social)](https://github.com/lmxhappy/step-by-step-learning-ai)

如果这个仓库对你有帮助，欢迎点击右上角 **Star ⭐** 支持一下，让更多人发现它！

欢迎来到 **StepByStepLearningAI** 仓库！这个仓库是微信公众号”稳扎稳打学AI”的配套资源库，旨在帮助大家逐步、系统地学习人工智能，特别是推荐算法领域。我们每天精挑细选最优质、最新鲜的推荐算法论文，陪你一起保持进步、不断精进。

## 关于公众号“稳扎稳打学AI”
- **公众号介绍**：关注我，每天为你精选推荐算法paper，帮助你稳扎稳打地学习AI。公众号由刘明星（微信: lmxhappy）运营，内容聚焦于推荐、广告和搜索等。
![e](imgs/qcode.jpg)
- **paper推荐原则**：我们优先选择高影响力、实用性强的论文，包括但不限于LLM在推荐中的应用、搜索词推荐等实际案例分析。
- **知乎专栏**：更多内容可查看[稳扎稳打学AI - 知乎专栏](https://zhuanlan.zhihu.com/stupid-ai)，已更新200+篇内容，涵盖热门论文解读。


## 仓库内容
这个仓库将逐步上传与公众号同步的资源，包括：
- **论文笔记和代码**：针对公众号推荐的论文，提供详细解读、代码实现和实验复现。
- **学习路径**：步步为营的AI学习指南，从基础到高级推荐算法。
- **示例文件**：
  - `MTFM.md`：更多技术细节和模型框架（More Technical Framework Markdown）。
- **未来计划**：TODO。

## 如何开始
1. **关注公众号**：搜索“稳扎稳打学AI”或扫描二维码，获取每日论文推荐。
2. **克隆仓库**：`git clone https://github.com/lmxhappy/StepByStepLearningAI.git`
3. **贡献**：欢迎提交PR，分享你的学习笔记或代码实现。我们一起进步！

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

- **[特征交叉](./feature-cross/)**
  - [MGDIN — 特征交叉新方法，CTR+3% (阿里)](./feature-cross/ali-mgdin.md)
  - [SlimPer — 多层多槽位 Target Attention 增强 (Meta)](./feature-cross/meta-slimper.md)
  - [DANet — 折扣率显式建模，GMV+2% (阿里)](./feature-cross/ali-danet.md)

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

- **[长尾问题](./long-tail/)**
  长尾样本、稀疏特征、低频物品、新用户、新item的建模与优化。
  - [UTTSI — 特征随机丢弃+集成解决稀疏样本预测不准 (阿里)](./long-tail/uttsi.md)

## 联系方式
- 作者：刘明星
- 微信：lmxhappy
- 知乎：[@思达-刘明星](https://www.zhihu.com/people/wang-wang-20-73)
- 反馈：欢迎在公众号留言或issue中讨论。

感谢你的关注！让我们稳扎稳打，学好AI！🚀