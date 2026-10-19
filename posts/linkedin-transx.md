![配图](../imgs/linkedin-transx.png)

---

LinkedIn's TransX: CTR +6.02% over a production HSTU-style model, by taking the user side off the request path.

The idea in one line: encode the user's history offline with a causal Transformer, compress it into 64 vectors stored as KV, and at request time only let each candidate attend to those 64.

Precomputing the user side is old. The compression layer is what's new: one vector loses detail, the full sequence KV grows with history, 64 vectors get both.

It's also strikingly close to Kuaishou's UxSID. They just compress differently.

Full breakdown (the three stages, why 64 vectors still see the whole history, and the UxSID comparison):
https://github.com/lmxhappy/step-by-step-learning-ai/blob/main/behavior-sequence-modeling/ultra-long-behavior-sequence-modeling/linkedin-transx.en.md

#RecommenderSystems #LinkedIn #MachineLearning #RecSys #AI
