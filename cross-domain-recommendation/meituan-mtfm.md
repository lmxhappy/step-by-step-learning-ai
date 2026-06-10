# 从MTGR到MTFM：美团异构多场景推荐Foundation Model

![](https://files.mdnice.com/user/171662/459564bf-7329-4064-848f-6e8ec67b8562.png)

**TLDR:** 工业推荐系统通常包含多个业务场景，但现有跨域推荐（CDR）与多场景推荐（MSR）方法往往需要高昂的训练资源，并依赖严格的输入对齐，限制了方法的可扩展性。同时，异构业务各自有着独立优化目标（例如抖音/小红书的短视频、直播、笔记），业务模型的独立优化迭代带来的数据隔离，影响了对于用户全场景、全生命周期的兴趣建模。

我们提出 **MTFM**（**M**ei**t**uan **F**oundation **M**odel for Recommendation），这是一个基于 Transformer 的统一框架，用于解决上述问题。MTFM不需所有场景特征对齐，而是将跨场景的用户行为数据转换为异构 Token，并通过统一的模型架构建模多场景知识。

为提升训推效率，我们首先提出多场景用户级样本聚合策略，通过显著减少样本行数来提升训练吞吐；同时结合 Grouped-Query Attention 与定制化 Hybrid Target Attention，降低显存开销与计算复杂度。

此外，我们在系统层面进行了多项优化（例如 Kernel 融合与消除 CPU-GPU 阻塞），进一步提升训练与推理吞吐。离线与在线实验验证了 MTFM 的有效性，结果表明随着模型规模与多场景数据规模提升，模型性能可持续增长。在线实验显示核心场景**订单量SQS +2.98\%, PHF +1.45\%**

论文地址：[https://arxiv.org/pdf/2602.11235](https://arxiv.org/pdf/2602.11235)

## 引言

在 Foundation Model 时代，大语言模型（LLM）的统一架构已从单模态发展到多模态。我们认为，如果模型仍被限制在单场景孤岛中，推荐基础模型的潜力无法真正释放。受多模态基础模型启发，推荐系统下一阶段的关键在于跨场景异构信息的统一建模。

我们认为，一个理想的推荐基础模型应具备三项核心属性：可扩展性（Scalability）、可拓展性（Extensibility）和高效率（Efficiency）。
第一，可扩展性要求模型在参数与数据规模增长时，性能能够稳定、可预期地提升。
第二，可拓展性要求模型能够低成本适配已有场景并接入新场景。
第三，高效率要求模型在多业务海量数据下，依然具备可接受的训练与推理成本。

现有多场景建模方法多遵循 “先同构化、再解耦（harmonize-then-decompose）” 范式：先将多场景特征对齐到固定模板，再通过参数解耦学习领域共性与特性。无法对齐的异构部分通常被丢弃或填充。模型结构上，这类方法常将参数拆为域共享与域特有部分（如基于 MoE 的结构。

然而，上述范式难以满足推荐基础模型要求：
1) **可拓展性不足**：工业场景间特征体系差异大，强行对齐会带来信息损失与维护困难。
2) **结构可扩展性不足**：大量依赖人工结构设计，难以验证在更大规模下的稳定收益。
3) **计算成本高**：传统范式下训练成本随数据量线性增长，多场景融合代价过高。

为解决这些问题，我们提出美团推荐基础模型 MTFM，统一满足上述三项属性。

在可拓展性方面，MTFM 以异构 Token 序列替代固定特征模板，使不同场景信号可在无需手工对齐的前提下统一输入。
在可扩展性方面，我们采用 Transformer 风格主干，利用自注意力机制自动学习跨场景共性与差异，减少启发式结构设计依赖。

在高效率方面，我们提出 Hybrid Target Attention（HTA）：在少量关键层使用 Full Attention 以保留全局建模能力，在多数层使用 Target Attention 降低复杂度。受 LLM 稀疏注意力思想启发，HTA 在效果与成本之间取得更优折中。结合 Grouped-Query Attention（GQA）后，可进一步降低显存并提升吞吐。

除模型设计外，我们还从数据与系统端协同优化。数据侧采用多场景用户级聚合，将离散样本转为高密度序列以提高训练效率；系统侧通过 CPU-GPU 管线编排、Kernel 融合、结构化稀疏等手段，显著提升训练与推理吞吐，使模型具备大规模工业落地可行性。

我们在美团多个场景进行了离线与在线实验。离线结果显示，MTFM 在多场景、多目标任务上持续优于主流基线：CTR 任务 GAUC 平均提升 0.36pp，最高 0.76pp；CTCVR 任务 GAUC 平均提升 0.29pp，最高 0.53pp。在线 A/B 测试中，神抢手（SQS）券包推荐订单提升 +2.98%，拼好饭（PHF）推荐订单提升 +1.45%。

本文贡献如下：
- **统一基础模型架构**：提出 MTFM，以异构 Token 化实现跨场景统一建模，兼顾可拓展性与可扩展性。
- **Hybrid Target Attention**：提出 HTA 并结合 GQA，在建模能力与计算效率间实现更优平衡。
- **系统-模型协同设计**：在数据、训练、推理链路做系统性优化，保障工业级经济可行性。
- **真实工业验证**：在美团离线与在线场景中取得显著增益，证明方法的业务价值。


## 方法


### 多场景数据下的用户样本聚合
本节介绍 MTFM 数据链路设计。沿用 MTGR 的用户级样本压缩，我们进一步扩展到多场景用户级压缩。

![](https://files.mdnice.com/user/171662/89e42821-d3ec-4837-8465-2c442a4a931d.png)

具体地，我们将特征拆分为场景无关共享特征（如 $H,R$）与场景特定特征（如 $U,C,I$）。先在各场景内对场景特定特征做用户级聚合，再按列拼接成统一表示，最后与共享特征按用户级合并。该设计有效减少重复计算与冗余存储，提升数据链路鲁棒性和资源利用效率。


### MTFM 模型架构


![](https://files.mdnice.com/user/171662/e28416f0-52b9-4f88-99bd-65981c2c77c4.png)


#### 异构 Token 化

我们将输入特征表示为可变长度的异构 Token 序列。

模型包含三类 Token：H-token、R-token、T-token。

以 $\{H_i\}$ 为例，序列中每个 item $h_{ij}$ 对应一个 H-token。先对原始特征做 Embedding，再通过 MLP 投影到统一维度 $d_{model}$：

$$
h_{ij} = MLP_i(Emb(h_{ij}))
$$

由于不同历史序列的特征异构，我们使用不同的 $MLP_i$ 作为对应 Tokenizer。

随后按时间顺序将所有历史序列 Token 合并，得到矩阵 $H \in R^{L_H \times d_{model}}$，其中 $L_H$ 是历史序列总长度。

同理，实时序列 Token 化为矩阵 $R \in R^{L_R \times d_{model}}$。

每次曝光对应一个 T-token，其输入由用户特征、交叉特征和目标物品特征拼接而成：

$$
t_{i}^s = MLP_{s} (Emb(U^s) \| Emb(C_i^s) \| Emb(I_i^s ))
$$

其中 $\|$ 为列拼接。所有曝光行为组成矩阵 $T \in R^{L_T \times d_{model}}$。

最终将 H/R/T 三类 Token 按行拼接，作为混合注意力架构输入：

$$
X^{(0)} = (H; R; T) \in R^{N \times d_{model}}
$$

其中 $(;)$ 为行拼接，$N = L_H+L_R+L_T$，不同用户的 $N$ 可不同。

#### Hybrid Target Attention 架构

Transformer架构很适合建模异构 Token 序列，但在多场景长序列下，$O(n^2)$ 复杂度带来显著计算瓶颈。为兼顾效果与效率，我们提出 Hybrid Target Attention：模型由 $B$ 个 Block 堆叠，每个 Block 由 1 层 Full Attention 与 $K$ 层 Target Attention 交替组成，在保留全局依赖建模能力的同时降低计算开销。
我们先介绍 Full Attention Layer 与 Target Attention Layer。
在 Full Attention Layer 中，先做分组 LayerNorm：不同历史序列 H-token 分组，不同实时序列 R-token 分组，不同场景 T-token 分组，以适配不同来源 Token 的分布差异：

$$
\widetilde{X}^{(l)} = GLN(X^{(l)} )
$$

归一化后送入 HSTU 建模，并使用 GQA 降低计算开销。
第 $l$ 层 Full Attention 可写为：
![](https://files.mdnice.com/user/171662/0d570307-3165-4220-909c-00e9a8506a69.png)
其中 $H$ 为 Query 头数，$G$ 为 KV 头数，$r=H/G$；$f_1^{(l)}, f_2^{(l)}$ 为线性层；$\phi_1, \phi_2$ 为非线性；$M \in R^{N \times N}$ 为动态 Mask，避免信息泄露。依据时间戳，Mask 构造规则为：
- H-token 对所有 Token 可见；
- R-token 仅对时间更晚的 Token 可见；
- T-token 仅对自身可见。
![](https://files.mdnice.com/user/171662/be1aaf46-dbe0-4733-a128-40ba7c4e73a3.png)
在 Target Attention Layer 中，仅更新 T-token，其他 Token 通过捷径连接传递到下一层。
先从归一化 Token 矩阵与动态 Mask 中取出 T-token 对应部分：

![](https://files.mdnice.com/user/171662/b1978f6b-04da-49b6-bd4c-af08d2d3a084.png)
随后用 HSTU+GQA 更新 T-token。第 $l$ 层 Target Attention 定义为：

![](https://files.mdnice.com/user/171662/14f2b591-ecf0-48dc-9d6e-224faa2d9924.png)
最后，我们将T-token的新embedding与其他token的上一层embedding进行拼接，从而生成target attention层的输出.通过这种基于Target Attention的混合架构，我们将计算资源优先分配给处理更关键的T-tokens，在显著降低计算资源消耗的同时确保性能不受影响。

![](https://files.mdnice.com/user/171662/610db714-99e3-496f-be4e-3f13d0b85032.png)
最后一层的T-tokens的embedding被输入到一个MMoE，来计算多场景不同目标的预估分.该混合架构将复杂度从 $O(N^2)$ 降至 $O(\frac{KNL_T + N^2}{K+1})$（其中 $L_T \ll N$）。实验表明 HTA 可在几乎无损效果下实现约 2 倍训练吞吐提升。
## 训练&推理优化
### 训练优化
在LLM领域，通过对文本的tokenization，特征均位于GPU上，训练高效。而在推荐系统中，特征工程繁重并且很多特征位于CPU上,这就导致了位于GPU上的模型和CPU上的特征之间存在阻塞(block)。由于特征处理、模型融合等环节涉及大量Host与Device之间的数据同步和串行依赖,流程中存在较多同步点,Host必须等待Device侧操作完成后才能继续执行,导致整体流程出现较多阻塞(Blocking)。这些CPU-GPU Pipeline Stall严重影响了整个训练性能。
1. 通过CUDA Profiler系统性地检查了训练框架中的CPU/GPU阻塞问题。优化策略主要包括两个方面:一是消除同步点,使CPU和GPU执行相互掩盖(overlap);二是优化Device端的内存操作,减少Device-to-Device(D2D)拷贝开销。 具体而言,我们针对频繁的张量索引赋值操作进行了D2D优化,将原本需要多次D2D拷贝的操作合并为单次原子操作。通过系统性地消除Pipeline阻塞点并优化D2D操作,最终实现了40%的训练吞吐提升。
2. 针对Flash Attention 2在自定义稀疏掩码场景下的性能瓶颈，我们提出了基于异步拷贝和共享内存流水线的优化方案。现有实现仅支持标准掩码，无法处理业务中的非标准稀疏掩码，且全局内存访问不连续导致效率低下。我们通过构造连续对齐的掩码内存布局满足异步拷贝要求，并在有限共享内存中精细设计数据流策略，即前向计算缓存掩码块，反向计算实现掩码与中间变量分时复用，从而将掩码加载延迟隐藏于计算流水线中。
3. 我们基于Triton框架实现了Group Layer Normalization（GLN）和动态掩码构建的融合算子。原生PyTorch实现采用Gather-Compute-Scatter模式处理GLN，并通过碎片化操作构造动态掩码，导致频繁的kernel启动开销和低效的内存访问模式。基于Triton框架，我们通过算子融合策略消除中间结果的反复读写，利用向量化内存访问提升带宽利用率，并设计分组并行计算模式以充分发挥GPU并行性。实验表明，融合算子相比原生实现显著降低了kernel启动开销和内存访问延迟，有效消除了训练流水线中的计算瓶颈。

### 推理优化
为了提升MTFM模型在大规模工业场景下的推理效率，我们从算子级、模型级到系统部署级实施了一系列优化策略：
1. 基于Ampere架构的结构化稀疏（Structured Sparsity）： 利用NVIDIA Ampere架构的稀疏张量核心（Sparse Tensor Cores），我们在HSTU组件的线性投影层（Q/K/V及输出层）实施了2:4结构化剪枝。该方法在保持模型精度的同时，实现了50%的显存压缩，并利用双倍理论峰值算力显著减少了矩阵乘法的计算延迟。
2. 细粒度注意力计算剪枝（Fine-grained Attention Pruning）： 针对动态掩码中的不规则稀疏性，我们设计了计算跳过机制。该机制不仅动态剔除序列填充（Padding）产生的无效计算，还结合业务先验知识（如屏蔽用户特征对目标特征的无效注意力权重），相比标准因果掩码显著降低了注意力机制的计算开销。
3. 场景化子图部署与系统级加速（Scenario-aware Deployment & Acceleration）： 由于MTFM是基于多个场景统一训练的，但是会被部署在各个独立子场景上，我们将完整计算图拆解为定制化子图以提升资源利用率，消除了本场景外的针对其他场景特征的额外计算。此外，结合BF16半精度推理与M-Falcon智能批处理算法，在平衡精度的前提下进一步最大化了推理吞吐量。

## 实验
MTFM在HP(美团首页)、PHF（拼好饭）、SQS（神抢手），分别进行商家、券包、菜品推荐数据上进行。这些场景在业务运营和推荐架构方面都持续优化了多年，是美团外卖生态体系内具有业务影响力的核心场景。
### 整体对比
![](https://files.mdnice.com/user/171662/226bf746-01c1-4641-931c-2734f404c46b.png)
首先，RankMixer 在通用推荐模型中表现有竞争力：在 HP 上略优于其他通用基线，在 SQS 上收益显著，相比 MMoE 在 CTCVR-GAUC 与 WRITE-GAUC 分别提升 0.38pp 与 0.49pp；但在 PHF 上虽优于 DCNv2，仍弱于 MMoE。

其次，STAR、PEPNet 等多场景方法通常优于通用基线，但整体仍弱于 MTGR、OneTrans 等生成式排序模型，并存在明显“跷跷板效应”。例如 PEPNet 在 HP、PHF 上优于 RankMixer，但在 SQS 上反而落后。

相比之下，生成式模型更全面，MTGR 在多数指标上略优于 OneTrans。

最终，MTFM 在几乎所有场景和任务上取得 SOTA。结果说明 MTFM 不仅缓解了“跷跷板效应”，更能高效利用多场景大数据，充分释放规模化收益。
### HTA 效率分析

![](https://files.mdnice.com/user/171662/be8755ff-22dd-4f2a-9cbf-6ca4d24e17d6.png)

不同 target/full 注意力比例配置的性能对比。

吞吐定义为单张 NVIDIA A100 GPU 上每秒处理样本数。我们在 7 天训练样本上系统比较了不同混合配置。将网络记为 $(K:P) \times B$，其中 $B$ 为 Block 数，$K$ 和 $P$ 分别是每个 Block 的 Target Attention 层数与 Full Attention 层数。我们也评估了纯 Target Attention（对应 OneRecV2 式 lazy decoder）。

结果表明，1:1 与 3:1 配置相较纯 Full Attention 基本无损甚至略优，同时显著提升吞吐并降低显存；当稀疏比例提升到 5:1 后性能出现约 0.07pp 回落，纯 Target Attention 回落约 0.12pp。引入 GQA 可进一步提升吞吐并降低显存。最终 HTA 相比 Full Attention 获得约 2 倍训练提速。

### Scalability

![](https://files.mdnice.com/user/171662/5221f839-0a28-4c9a-a0fb-0c93dc98ed2c.png)
上图(a)中，我们以MMoE的计算量（GFLOPs）为基准，在不同容量模型下（从10x 到 70x）进行实验，结果显示MTFM在不同场景中的CTCVR GAUC中展现出稳定的缩放定律，说明其在多场景推荐（MSR）任务中关于模型容量的可扩展性。

上图(b)展示了三种不同模型规模在持续训练过程中，随训练 Token 增长的 SQS-CTCVR GAUC 变化。结果表明，所有模型均随数据增长稳定提升；在更大数据规模下，不同模型尺寸间性能差距会逐步拉开，说明 MTFM 能稳定将更多训练数据转化为性能收益。

### 可解释性分析

![](https://files.mdnice.com/user/171694/9fda1cf5-21a1-4aca-9628-320bb6b0a9e0.jpg)
同时我们对与MTFM进行了可解释性分析，将模型最后一层的cross attention分数进行了可视化，可以发现不同场景的targets在不同的行为序列中有明显的场景区分性（具体说明可参考原论文），说明MTFM架构能够多场景数据中学习，具备场景感知能力。

### 在线实验

为验证真实业务价值，我们在两个核心线上场景开展 A/B 测试。实验流量覆盖千万级日曝光，基线为已长期迭代上线的 SOTA 模型。


![](https://files.mdnice.com/user/171662/dc1831dd-963e-4e34-829c-6449f1cc352c.png)


如上所示，MTFM 在关键业务指标上均取得稳定正向收益。

在该业务中，线上订单提升通常需要多轮模型迭代累计；本次实验中的增益相当于 2-3 轮迭代效果总和。同时，系统优化也带来更低线上延迟。

上述结果表明，MTFM 能在真实工业推荐系统中带来直接业务价值。

## 结论

本文提出 MTFM：一种面向推荐系统的基础模型，通过异构 Token 化实现真正免对齐的跨域统一建模。通过算法、训练与部署的Co-design，MTFM 同时满足可扩展性、可拓展性与高效率三项核心属性。大量离线实验表明其显著优于现有方法，构建了突破单场景范式的新路径；线上 A/B 测试进一步验证其在美团真实业务中的显著收益。
