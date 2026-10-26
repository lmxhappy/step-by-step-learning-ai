![配图](../imgs/linkedin-pair.png)

---

LinkedIn's PAIR: CTR +7.5% and Premium retention +2.1% for people recommendation, with no new technique in it.

It's a two-tower retriever with a 0.6B LLM encoder, a weighted average over the user's last 10 clicks, and exposed-but-unclicked plus random negatives. All familiar.

What actually moved the numbers: production used LinkedIn's platform-wide general-purpose member embedding, never trained for this surface. Fine-tuning on this surface's own clicks did most of the work. Customized beats general-purpose, almost every time.

One thing I'd push back on: the paper keeps saying professional relevance is "complementary" (job seekers want recruiters, not peers), but nothing in the model targets that, and no result shows it.

Full breakdown:
https://github.com/lmxhappy/step-by-step-learning-ai/blob/main/deep-retrieval/linkedin-pair.en.md

#RecommenderSystems #LinkedIn #MachineLearning #InformationRetrieval #AI
