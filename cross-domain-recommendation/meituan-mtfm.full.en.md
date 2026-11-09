# From MTGR to MTFM: Meituan's foundation model for heterogeneous multi-scenario recommendation

> [中文版](meituan-mtfm.md)

![](https://files.mdnice.com/user/171662/459564bf-7329-4064-848f-6e8ec67b8562.png)

**TLDR:** Industrial recommender systems usually span many business scenarios, but existing cross-domain recommendation (CDR) and multi-scenario recommendation (MSR) methods often need expensive training resources and depend on strict input alignment, which limits how far they scale. At the same time, heterogeneous businesses each have their own optimization objectives (e.g. short video, live streaming and notes on Douyin / Xiaohongshu), and the data silos created by optimizing and iterating each business model independently hurt modeling of user interests across all scenarios and the whole user lifecycle.

We propose **MTFM** (**M**ei**t**uan **F**oundation **M**odel for Recommendation), a unified Transformer-based framework to address these problems. MTFM does not require features to be aligned across scenarios. Instead, it converts cross-scenario user behavior data into heterogeneous tokens and models multi-scenario knowledge with one unified architecture.

To improve training and inference efficiency, we first propose a multi-scenario user-level sample aggregation strategy, which raises training throughput by greatly reducing the number of sample rows; we also combine Grouped-Query Attention with a customized Hybrid Target Attention to cut GPU memory and compute complexity.

In addition, we made several system-level optimizations (e.g. kernel fusion and removing CPU-GPU blocking) that further raise training and inference throughput. Offline and online experiments confirm MTFM's effectiveness, and show that performance keeps improving as model size and multi-scenario data size grow. Online, the core scenarios show **orders: SQS +2.98%, PHF +1.45%**.

Paper: [https://arxiv.org/pdf/2602.11235](https://arxiv.org/pdf/2602.11235)

## Introduction

In the foundation model era, the unified architecture of large language models (LLMs) has evolved from single-modal to multimodal. We believe that if a model stays confined to single-scenario silos, the potential of recommendation foundation models cannot really be unlocked. Inspired by multimodal foundation models, the key to the next stage of recommender systems is unified modeling of heterogeneous information across scenarios.

We believe an ideal recommendation foundation model should have three core properties: scalability, extensibility, and efficiency.
First, scalability requires that performance improve steadily and predictably as parameters and data grow.
Second, extensibility requires that the model adapt to existing scenarios and onboard new ones at low cost.
Third, efficiency requires that training and inference cost stay acceptable under massive multi-business data.

Existing multi-scenario modeling methods mostly follow a "harmonize-then-decompose" paradigm: first align multi-scenario features to a fixed template, then learn domain-shared and domain-specific patterns via parameter decoupling. Heterogeneous parts that cannot be aligned are usually dropped or padded. Architecturally, these methods often split parameters into domain-shared and domain-specific parts (e.g. MoE-based structures).

However, this paradigm struggles to meet the requirements of a recommendation foundation model:
1) **Insufficient extensibility**: feature systems differ a lot across industrial scenarios, and forcing alignment causes information loss and maintenance pain.
2) **Insufficient architectural scalability**: heavy reliance on hand-designed structures makes it hard to verify stable gains at larger scale.
3) **High compute cost**: under the traditional paradigm, training cost grows linearly with data volume, making multi-scenario fusion too expensive.

To solve these problems, we propose MTFM, Meituan's recommendation foundation model, which meets all three properties.

For extensibility, MTFM replaces fixed feature templates with heterogeneous token sequences, so signals from different scenarios can be fed in together without manual alignment.
For scalability, we adopt a Transformer-style backbone and let self-attention learn cross-scenario commonalities and differences automatically, reducing reliance on heuristic structure design.

For efficiency, we propose Hybrid Target Attention (HTA): use Full Attention in a few key layers to keep global modeling ability, and Target Attention in most layers to reduce complexity. Inspired by sparse attention in LLMs, HTA strikes a better trade-off between quality and cost. Combined with Grouped-Query Attention (GQA), it further cuts GPU memory and raises throughput.

Beyond model design, we also co-optimize the data and system sides. On the data side, multi-scenario user-level aggregation turns scattered samples into dense sequences to improve training efficiency; on the system side, CPU-GPU pipeline orchestration, kernel fusion, structured sparsity and similar techniques substantially raise training and inference throughput, making large-scale industrial deployment feasible.

We ran offline and online experiments in several Meituan scenarios. Offline, MTFM consistently beats mainstream baselines on multi-scenario, multi-objective tasks: on CTR tasks GAUC improves by 0.36pp on average and up to 0.76pp; on CTCVR tasks GAUC improves by 0.29pp on average and up to 0.53pp. In online A/B tests, orders from Shenqiangshou (SQS) coupon-package recommendation rose +2.98%, and orders from Pinhaofan (PHF) recommendation rose +1.45%.

Contributions:
- **Unified foundation model architecture**: we propose MTFM, which achieves unified cross-scenario modeling via heterogeneous tokenization, delivering both extensibility and scalability.
- **Hybrid Target Attention**: we propose HTA and combine it with GQA for a better balance between modeling power and compute efficiency.
- **System-model co-design**: systematic optimization across the data, training and inference pipelines ensures industrial-grade economic feasibility.
- **Real industrial validation**: significant gains in Meituan's offline and online scenarios demonstrate the method's business value.

## Method

### User sample aggregation over multi-scenario data
This section describes MTFM's data pipeline design. Building on MTGR's user-level sample compression, we extend it to multi-scenario user-level compression.

![](https://files.mdnice.com/user/171662/89e42821-d3ec-4837-8465-2c442a4a931d.png)

Specifically, we split features into scenario-agnostic shared features (e.g. $H,R$) and scenario-specific features (e.g. $U,C,I$). Scenario-specific features are first aggregated at user level within each scenario, then concatenated column-wise into a unified representation, and finally merged with the shared features at user level. This design effectively reduces repeated computation and redundant storage, and improves data pipeline robustness and resource utilization.

### MTFM model architecture

![](https://files.mdnice.com/user/171662/e28416f0-52b9-4f88-99bd-65981c2c77c4.png)

#### Heterogeneous tokenization

We represent the input features as a variable-length sequence of heterogeneous tokens.

The model has three token types: H-tokens, R-tokens and T-tokens.

Take $\{H_i\}$ as an example: each item $h_{ij}$ in the sequence corresponds to one H-token. The raw features are first embedded, then projected by an MLP to a unified dimension $d_{model}$:

$$
h_{ij} = MLP_i(Emb(h_{ij}))
$$

Since features differ across historical sequences, we use a different $MLP_i$ as the tokenizer for each.

All historical sequence tokens are then merged in time order into a matrix $H \in R^{L_H \times d_{model}}$, where $L_H$ is the total historical sequence length.

Likewise, the real-time sequences are tokenized into a matrix $R \in R^{L_R \times d_{model}}$.

Each exposure corresponds to one T-token, whose input is the concatenation of user features, cross features and target item features:

$$
t_{i}^s = MLP_{s} (Emb(U^s) \| Emb(C_i^s) \| Emb(I_i^s ))
$$

where $\|$ is column-wise concatenation. All exposures form a matrix $T \in R^{L_T \times d_{model}}$.

Finally, the H/R/T tokens are concatenated row-wise as input to the hybrid attention architecture:

$$
X^{(0)} = (H; R; T) \in R^{N \times d_{model}}
$$

where $(;)$ is row-wise concatenation and $N = L_H+L_R+L_T$; $N$ can differ across users.

#### Hybrid Target Attention architecture

The Transformer architecture suits heterogeneous token sequences well, but with long multi-scenario sequences its $O(n^2)$ complexity becomes a serious compute bottleneck. To balance quality and efficiency, we propose Hybrid Target Attention: the model stacks $B$ blocks, each made of 1 Full Attention layer alternating with $K$ Target Attention layers, which keeps the ability to model global dependencies while lowering compute cost.
We first introduce the Full Attention layer and the Target Attention layer.
In the Full Attention layer, grouped LayerNorm is applied first: H-tokens from different historical sequences are grouped, R-tokens from different real-time sequences are grouped, and T-tokens from different scenarios are grouped, to handle the distribution differences across token sources:

$$
\widetilde{X}^{(l)} = GLN(X^{(l)} )
$$

After normalization, the tokens are fed into HSTU, with GQA used to cut compute cost.
Full Attention at layer $l$ can be written as:
![](https://files.mdnice.com/user/171662/0d570307-3165-4220-909c-00e9a8506a69.png)
where $H$ is the number of query heads, $G$ the number of KV heads, $r=H/G$; $f_1^{(l)}, f_2^{(l)}$ are linear layers; $\phi_1, \phi_2$ are nonlinearities; $M \in R^{N \times N}$ is a dynamic mask that prevents information leakage. Based on timestamps, the mask is built as follows:
- H-tokens are visible to all tokens;
- R-tokens are visible only to tokens later in time;
- T-tokens are visible only to themselves.
![](https://files.mdnice.com/user/171662/be1aaf46-dbe0-4733-a128-40ba7c4e73a3.png)
In the Target Attention layer, only T-tokens are updated; the other tokens are passed to the next layer through a shortcut connection.
First, the parts corresponding to T-tokens are taken from the normalized token matrix and the dynamic mask:

![](https://files.mdnice.com/user/171662/b1978f6b-04da-49b6-bd4c-af08d2d3a084.png)
Then HSTU+GQA is used to update the T-tokens. Target Attention at layer $l$ is defined as:

![](https://files.mdnice.com/user/171662/14f2b591-ecf0-48dc-9d6e-224faa2d9924.png)
Finally, the new T-token embeddings are concatenated with the other tokens' embeddings from the previous layer to produce the Target Attention layer's output. With this hybrid architecture based on Target Attention, compute is allocated first to the more important T-tokens, greatly reducing resource consumption while keeping performance intact.

![](https://files.mdnice.com/user/171662/610db714-99e3-496f-be4e-3f13d0b85032.png)
The last layer's T-token embeddings are fed into an MMoE to compute prediction scores for the different objectives of each scenario. The hybrid architecture reduces complexity from $O(N^2)$ to $O(\frac{KNL_T + N^2}{K+1})$ (where $L_T \ll N$). Experiments show HTA gives about 2x training throughput with almost no loss in quality.
## Training & inference optimization
### Training optimization
In the LLM world, text is tokenized and all features live on the GPU, so training is efficient. In recommender systems, feature engineering is heavy and many features live on the CPU, which causes blocking between the model on the GPU and the features on the CPU. Because feature processing, model fusion and similar steps involve lots of host-device data synchronization and serial dependencies, the pipeline has many sync points where the host must wait for device-side operations to finish before continuing, causing frequent blocking. These CPU-GPU pipeline stalls seriously hurt overall training performance.
1. We systematically examined CPU/GPU blocking in the training framework with the CUDA Profiler. The optimization strategy has two parts: first, remove sync points so CPU and GPU execution overlap; second, optimize device-side memory operations to reduce Device-to-Device (D2D) copy overhead. Specifically, we applied D2D optimization to frequent tensor index-assignment operations, merging operations that used to need several D2D copies into a single atomic operation. By systematically removing pipeline blocking points and optimizing D2D operations, we achieved a 40% increase in training throughput.
2. For Flash Attention 2's performance bottleneck with custom sparse masks, we propose an optimization based on asynchronous copies and a shared-memory pipeline. The existing implementation only supports standard masks, cannot handle the non-standard sparse masks in our business, and its non-contiguous global memory access is inefficient. We build a contiguous, aligned mask memory layout to meet the requirements of asynchronous copies, and carefully design the data-flow strategy within limited shared memory: the forward pass caches mask blocks, and the backward pass time-shares memory between masks and intermediate variables, hiding mask-loading latency inside the compute pipeline.
3. We implemented a fused operator for Group Layer Normalization (GLN) and dynamic mask construction in Triton. The native PyTorch implementation handles GLN with a Gather-Compute-Scatter pattern and builds the dynamic mask through fragmented operations, causing frequent kernel launch overhead and inefficient memory access. In Triton, we use operator fusion to avoid repeatedly reading and writing intermediate results, vectorized memory access to raise bandwidth utilization, and a grouped parallel compute pattern to fully use GPU parallelism. Experiments show the fused operator significantly reduces kernel launch overhead and memory access latency compared with the native implementation, effectively removing the compute bottleneck in the training pipeline.

### Inference optimization
To improve MTFM's inference efficiency in large-scale industrial scenarios, we applied a series of optimizations at the operator, model and deployment levels:
1. Structured sparsity on the Ampere architecture: using the Sparse Tensor Cores of NVIDIA's Ampere architecture, we applied 2:4 structured pruning to the linear projection layers (Q/K/V and output) of the HSTU component. While preserving model accuracy, this gives 50% GPU memory compression, and the doubled theoretical peak compute significantly reduces matrix-multiplication latency.
2. Fine-grained attention pruning: for the irregular sparsity in the dynamic mask, we designed a compute-skipping mechanism. It not only dynamically drops wasted compute from sequence padding, but also uses business prior knowledge (e.g. masking out ineffective attention weights from user features to target features), significantly lowering attention compute compared with a standard causal mask.
3. Scenario-aware deployment & acceleration: MTFM is trained jointly over many scenarios but deployed in each separate sub-scenario, so we split the full compute graph into customized subgraphs to improve resource utilization, removing extra compute on other scenarios' features that a given scenario does not need. In addition, BF16 half-precision inference combined with the M-Falcon smart batching algorithm further maximizes inference throughput while keeping accuracy balanced.

## Experiments
MTFM was evaluated on HP (Meituan homepage), PHF (Pinhaofan) and SQS (Shenqiangshou), using merchant, coupon-package and dish recommendation data respectively. These scenarios have been continuously optimized for years in both business operations and recommendation architecture, and are core, business-critical scenarios in the Meituan food delivery ecosystem.
### Overall comparison
![](https://files.mdnice.com/user/171662/226bf746-01c1-4641-931c-2734f404c46b.png)
First, RankMixer is competitive among general recommendation models: on HP it is slightly better than other general baselines, and on SQS the gain is significant, improving CTCVR-GAUC and WRITE-GAUC by 0.38pp and 0.49pp over MMoE; but on PHF, although it beats DCNv2, it is still weaker than MMoE.

Second, multi-scenario methods such as STAR and PEPNet usually beat the general baselines, but overall remain weaker than generative ranking models such as MTGR and OneTrans, and show a clear "seesaw effect". For example, PEPNet beats RankMixer on HP and PHF but falls behind on SQS.

By contrast, generative models are more well-rounded, and MTGR is slightly better than OneTrans on most metrics.

In the end, MTFM achieves SOTA on almost all scenarios and tasks. This shows MTFM not only mitigates the "seesaw effect" but also uses large multi-scenario data efficiently, fully unlocking the gains from scale.
### HTA efficiency analysis

![](https://files.mdnice.com/user/171662/be8755ff-22dd-4f2a-9cbf-6ca4d24e17d6.png)

Performance comparison across different target/full attention ratio configurations.

Throughput is defined as samples processed per second on a single NVIDIA A100 GPU. We systematically compared different hybrid configurations on 7 days of training samples. We denote a network as $(K:P) \times B$, where $B$ is the number of blocks, and $K$ and $P$ are the numbers of Target Attention layers and Full Attention layers per block. We also evaluated pure Target Attention (corresponding to a OneRecV2-style lazy decoder).

Results show that the 1:1 and 3:1 configurations are essentially lossless or even slightly better than pure Full Attention, while significantly raising throughput and lowering GPU memory; when the sparsity ratio goes up to 5:1, performance drops by about 0.07pp, and pure Target Attention drops by about 0.12pp. Adding GQA further raises throughput and lowers GPU memory. In the end, HTA trains about 2x faster than Full Attention.

### Scalability

![](https://files.mdnice.com/user/171662/5221f839-0a28-4c9a-a0fb-0c93dc98ed2c.png)
In (a) above, taking MMoE's compute (GFLOPs) as the baseline, we ran experiments with models of different capacity (from 10x to 70x). MTFM shows a stable scaling law in CTCVR GAUC across scenarios, demonstrating its scalability with respect to model capacity on multi-scenario recommendation (MSR) tasks.

(b) above shows how SQS-CTCVR GAUC changes as training tokens grow during continued training, for three model sizes. All models improve steadily as data grows; at larger data scales the performance gap between model sizes gradually widens, showing that MTFM can reliably turn more training data into performance gains.

### Interpretability analysis

![](https://files.mdnice.com/user/171694/9fda1cf5-21a1-4aca-9628-320bb6b0a9e0.jpg)
We also did an interpretability analysis of MTFM, visualizing the cross-attention scores of the model's last layer. Targets from different scenarios show clear scenario-specific patterns over the different behavior sequences (see the original paper for details), indicating that the MTFM architecture can learn from multi-scenario data and is scenario-aware.

### Online experiments

To verify real business value, we ran A/B tests in two core online scenarios. Experiment traffic covered tens of millions of daily exposures, and the baseline was a SOTA model that had been iterated and serving online for a long time.

![](https://files.mdnice.com/user/171662/dc1831dd-963e-4e34-829c-6449f1cc352c.png)

As shown above, MTFM delivers stable positive gains on key business metrics.

In this business, online order gains usually accumulate over several rounds of model iteration; the gain in this experiment equals the total of 2-3 rounds of iteration. The system optimizations also brought lower online latency.

These results show MTFM can deliver direct business value in a real industrial recommender system.

## Conclusion

This paper proposes MTFM, a foundation model for recommender systems that achieves truly alignment-free unified cross-domain modeling through heterogeneous tokenization. Through co-design of algorithms, training and deployment, MTFM satisfies all three core properties: scalability, extensibility and efficiency. Extensive offline experiments show it significantly outperforms existing methods, opening a new path beyond the single-scenario paradigm; online A/B tests further confirm its significant gains in Meituan's real business.
