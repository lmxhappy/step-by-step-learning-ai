# YouTube短视频模型，增加用户SID特征，证明有用

关注我，每天为你精挑细选最优质、最新鲜的推荐算法paper，陪你一起保持进步、不断精进！

### 论文：TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems
### 网址：https://arxiv.org/abs/2606.25147
### 公司：Google DeepMind / YouTube
### 思想：生成式用户建模
### 方向：用户行为序列建模

## 解读：
本文是行为序列建模，核心目标是获得高质量的用户表征。用一个 Encoder-Decoder 模型同时输出离散用户 Token（SID 序列）和稠密用户 Embedding，两路表征分别或联合注入下游排序模型，实现用户兴趣的语义化、泛化性建模。其中，encoder对用户行为序列获得hidden states，decoder借助这些states生成用户SID。与此同时，states聚合成一个embed，即稠密用户Embedding。

![TokenMinds 整体架构（Figure 1）](../imgs/tokenminds-framework.png)

### Encoder（370M MoE）
处理完整用户历史序列，输出稠密用户 Embedding。其中，序列经过Encoder获得的hidden state seq，通过 last-token / mean pooling 等方式提取，获得维度 1152的一个embed，作为稠密用户 Embedding。

**输入序列构造：**
用户按时间顺序的 watch 序列（最近 1200 个）。每个 watch = <Condition> + prefix-L SID（L=4，粗粒度前缀） + 软 token（watch time ratio、device 等特征）。其中，<Condition> 是场景信息，包括 LFV（长视频）和 SFV（短视频）。序列里还包含了用户文本搜索行为。

其中，训练一个RQ-VAE获取item的SID，层数该RQ-VAE的时候，长度是$L_{full}$。但是在输入Encoder-Decoder 架构，生成离散的 SID 用户 Token的时候，只用前面的$L$层。Loss也只在 prefix-L SID 上计算。

**跨场景统一建模**：同时存在 LFV（长视频）和 SFV（短视频），两者消费模式差异大（SFV 是连续浏览反馈更强），但用户兴趣有重叠（前两个 prefix SID 重叠约 40%）。在每个 watch 前加场景 condition token（<LFV> / <SFV>）。共享同一个 SID 词表。

为什么SID训长用短？
如果用完整的SID，模型很容易记住具体视频，导致生成结果多样性差、泛化能力弱。使用较短的 prefix，能让模型在更粗的语义层次上进行生成，更容易产生多样化的兴趣 Token。

### Decoder（370M Dense）
自回归生成离散 SID 用户 Token（用 beam search 生成多条候选序列）。每个用户样本，不是预测下一个 watch，而是从未来 24h 窗口随机采样最多 N=15 个 target。一个用户样本（一个训练 example），Encoder：只跑 1 次 前向传播；Decoder：虽然要为 N 个 target 生成 SID，但不是简单地跑 N 次，而是共享同一个 Encoder 的 hidden states，通过 cross-attention 同时处理这 N 个目标。

为什么这样做？让模型学习近未来一段时间内的兴趣分布，而不是只盯住下一个瞬间行为。显著提升泛化能力，尤其是 cold-start 场景。

还有2个小技巧：
* 样本加权：每个用户样本，加入 engagement reward $r(W_i)$，鼓励高质量、多样消费。
* 重要性采样：按照 reward 大小对训练目标进行重要性采样，让高价值样本在训练中出现得更频繁。


### 补充——用户SID转成embed的3种方法：
Decoder 生成 B=40 条 SID 序列，需转换为可注入排序模型的 Embedding：
1. **Prefix Embedding Mapping**（静态）：将预测出的 prefix-L SID 映射回原始内容 Embedding，取该前缀下所有视频 Embedding 的均值。直觉清晰但缺乏学习能力。
2. **N-gram Embedding**（可学习）：将 SID 序列切成固定长度 n-gram，每个 n-gram 查可学习 Embedding Table，累加得到用户向量。每个 n-gram 对应一个可学习的 Embedding（从头训练的 Embedding Table），把这些 n-gram 的 Embedding 相加（或拼接），得到最终的用户向量。
3. **SentencePiece Embedding**（可学习）：对 SID 序列做变长子词分词（BPE 风格），查可学习 Embedding。用 SentencePiece（类似 BPE 子词切分） 对 SID 序列进行变长子词切分。SentencePiece 会从大量 SID 序列中学习出哪些 codeword 组合应该被当作一个“子词”。每个学到的子词都有一个可学习的 Embedding。最后把这些子词 Embedding 组合起来得到用户向量。

### A/B：
短视频，参与用户提升 +0.11%、满意互动提升 +0.62%。

## 心得：
* Item SID 是通过内容量化模型（RQ-VAE）“离散化”得到的语义 ID，而 User SID 是通过生成模型自回归“生成”出来的兴趣 Token。两者虽然都叫 SID，但本质上是两种不同范式的产物：一个是内容表征的离散化，一个是用户未来行为的生成式建模。
* 生成式用户建模——以预测用户未来消费的 item 语义 ID（SID）为训练目标，驱动 Encoder 学习具备预测性的用户表征。
* **双输出设计是核心亮点**：稠密 Embedding 捕捉连续兴趣空间，离散 Token 捕捉语义类别，两者天然互补——前者适合 retrieval 侧相似度计算，后者适合 ranking 侧特征交叉，联合注入效果最优。
* **SID 截断是反直觉的关键技巧**：用粗粒度前缀而非完整 SID 作为生成目标，以精度换泛化，cold-start 提升最为显著，去掉后下降高达 17.1%。

## 愚见
* Look-ahead 24h 窗口对时间敏感的场景（热点新闻、赛事直播）可能引入噪声，动态调整窗口大小值得探索。
* 生成模型，训练和预估不一致，预估的时候，没有target的相关非SID信息了。可能这个信息重要性不大，所以影响可控。
* 本文只用行为序列生成 User SID。但用户还有很多其他特征（人口属性、地域、设备、实时上下文等），这些特征同样携带兴趣信号。理论上可以把所有用户特征（包括行为序列）一起输入 Encoder，生成更全面的 User SID。这样 User SID 就不再只是"行为序列的压缩"，而是"用户整体画像的生成式表达"。

## 可信度：生产

## 推荐等级：有实践价值


**请帮忙点赞、转发，谢谢。欢迎干货投稿 \ 论文宣传\ 合作交流**


### 【铁粉】请入微信群，群内我会给出更深入的解读，还可以共同讨论技术方案、发招聘广告、内推和交友等。
* 铁粉标准：关注公众号一个月以上，且在公众号上累计15次互动（评论、爱心、转发）、或投稿1次、或打赏199，只欢迎技术同学。
* 入群方法：请您加个人微信lmxhappy，我拉您入群，请备注【公司】（只我个人看，不公开）。

## 推荐您继续阅读：
* TIGER / PLUM：生成式检索 SID 的前置工作，TokenMinds 是其在用户建模侧的延伸
* RQ-VAE for Recommendations：SID 生成方法的原始论文
* HSTU（Hierarchical Sequential Transduction Units）：Meta 发布的万亿参数级生成式推荐序列建模工作，与 TokenMinds 同属大模型做用户序列建模的方向

