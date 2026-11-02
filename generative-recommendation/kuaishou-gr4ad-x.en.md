![配图](../imgs/kuaishou-gr4ad-x.png)

---

Kuaishou's GR4AD-X: +3.07% ad revenue on top of a generative ad retriever that was already fully deployed.

GR4AD generates an ad's Semantic ID token by token from the user's sequence, then maps it back to an ad. GR4AD-X keeps that framework and fixes three things it found in production data:

1. Ads sitting between two first-level clusters get pruned by beam search almost at random. Fix: give them two first-level codes. Same beam, same latency, +0.58%.
2. Growing the backbone 3x costs two thirds of QPS for +0.53%. Fix: add capacity where serving is cheap, e.g. prefix lookup tables that replace an FFN gate.
3. Top-10 by probability and top-10 by eCPM barely overlap. Fix: per-prefix value heads that vote in beam search, plus a one-step rerank on the same network.

No new paradigm. Careful, data-driven polishing of every component, and a reminder of how much is left in a system after it ships.

Full breakdown:
https://github.com/lmxhappy/step-by-step-learning-ai/blob/main/generative-recommendation/kuaishou-gr4ad-x.full.en.md

#RecommenderSystems #GenerativeRecommendation #Advertising #MachineLearning #AI
