<p align="center">
  <img src="imgs/qcode.jpg" alt="稳扎稳打学AI" width="200"/>
</p>

<h1 align="center">📚 Step-by-Step Learning AI</h1>

<p align="center">
  <strong>每日精选推荐算法论文 · 稳扎稳打，从入门到前沿</strong>
</p>

<p align="center">
  <a href="https://zhuanlan.zhihu.com/stupid-ai"><img src="https://img.shields.io/badge/知乎专栏-200%2B%20篇-blue?logo=zhihu&logoColor=white" alt="知乎"/></a>
  <a href="#关注公众号"><img src="https://img.shields.io/badge/微信公众号-稳扎稳打学AI-07C160?logo=wechat&logoColor=white" alt="WeChat"/></a>
  <a href="https://github.com/lmxhappy/step-by-step-learning-ai/stargazers"><img src="https://img.shields.io/github/stars/lmxhappy/step-by-step-learning-ai?style=social" alt="Stars"/></a>
  <a href="https://github.com/lmxhappy/step-by-step-learning-ai/fork"><img src="https://img.shields.io/github/forks/lmxhappy/step-by-step-learning-ai?style=social" alt="Forks"/></a>
</p>

---

## ✨ 项目简介

本仓库是微信公众号 **「稳扎稳打学AI」** 的配套资源库，聚焦于**推荐系统、计算广告、搜索**等工业级AI领域。我们每天精选最优质的论文，提供深度解读与笔记，帮助你系统性地追踪学术前沿。

**🎯 内容特色**
- 聚焦工业界一线论文（字节、阿里、美团、Meta、Google 等）
- 覆盖推荐全链路：召回 → 粗排 → 精排 → 重排
- 兼顾经典方法论和最新趋势（LLM for Rec、生成式推荐、Scaling Law 等）
- 每篇笔记提炼核心贡献 + 方法对比 + 个人点评

## 📂 目录结构

```
step-by-step-learning-ai/
├── item-to-item/                    # I2I 推荐（Item-to-Item）
├── generative-recommendation/       # 生成式推荐
│   ├── OneLive.md
│   ├── Sigma.md
│   └── VectorizingTrie.md
├── behavior-sequence-modeling/      # 用户行为序列建模
│   └── ultra-long-behavior-sequence-modeling/
│       ├── stca.md                  # STCA: 万级序列端到端建模
│       └── Ultra-HSTU.md
├── feature-cross/                   # 特征交叉
│   └── MGDIN.md
├── cross-domain-recommendation/     # 跨域推荐
│   └── MTFM.md
├── SID/                             # Semantic ID / 语义索引
└── imgs/
```

## 📖 论文笔记索引

### 🔁 生成式推荐 (Generative Recommendation)

| 论文 | 笔记链接 | 关键词 |
|------|----------|--------|
| OneLive | [OneLive.md](generative-recommendation/OneLive.md) | 直播推荐、生成式检索 |
| Sigma | [Sigma.md](generative-recommendation/Sigma.md) | 语义生成 |
| VectorizingTrie | [VectorizingTrie.md](generative-recommendation/VectorizingTrie.md) | Trie向量化、索引结构 |

### 🧠 用户行为序列建模 (Behavior Sequence Modeling)

| 论文 | 笔记链接 | 关键词 |
|------|----------|--------|
| STCA | [stca.md](behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/stca.md) | 超长序列、端到端、抖音 |
| Ultra-HSTU | [Ultra-HSTU.md](behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/Ultra-HSTU.md) | 超长序列、HSTU扩展 |

### ⚡ 特征交叉 (Feature Interaction)

| 论文 | 笔记链接 | 关键词 |
|------|----------|--------|
| MGDIN | [MGDIN.md](feature-cross/MGDIN.md) | 多粒度交互 |

### 🌐 跨域推荐 (Cross-Domain Recommendation)

| 论文 | 笔记链接 | 关键词 |
|------|----------|--------|
| MTFM | [MTFM.md](cross-domain-recommendation/MTFM.md) | 多任务迁移 |

### 🆔 Semantic ID

详见 [SID/](SID/) 目录。

### 🔗 Item-to-Item 推荐

详见 [item-to-item/](item-to-item/) 目录。

> 📌 **持续更新中** — 更多论文笔记将随公众号内容同步上传。

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/lmxhappy/step-by-step-learning-ai.git

# 按主题浏览
cd step-by-step-learning-ai/generative-recommendation
```

## 🤝 参与贡献

欢迎通过以下方式参与：
1. ⭐ **Star** 本仓库，让更多人发现
2. 🍴 **Fork** 并提交 PR，分享你的论文笔记或代码实现
3. 💬 在 [Issues](https://github.com/lmxhappy/step-by-step-learning-ai/issues) 中讨论或建议新的论文主题

## 📬 关注公众号

搜索 **「稳扎稳打学AI」** 或扫码关注，获取每日推荐算法论文精选推送：

<p align="center">
  <img src="imgs/qcode.jpg" alt="公众号二维码" width="200"/>
</p>

## 📫 联系方式

- **作者**：刘明星
- **微信**：lmxhappy
- **知乎**：[@思达-刘明星](https://www.zhihu.com/people/wang-wang-20-73)
- **知乎专栏**：[稳扎稳打学AI](https://zhuanlan.zhihu.com/stupid-ai)（200+ 篇论文解读）

---

<p align="center">
  如果觉得有帮助，请给个 ⭐ Star，这是对我最大的支持！
</p>
