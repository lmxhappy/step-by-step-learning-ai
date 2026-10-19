![配图](../imgs/linkedin-transx.png)

---

Long-sequence recommenders re-encode the user's history on every request. LinkedIn's TransX takes the user side off the request path entirely.

Paper: TransX Technical Report, LinkedIn (link in the first comment)

The pipeline has three stages:
behavior sequence -> (1) causal Transformer -> (2) compress to 64 vectors, stored as KV -> (3) candidate cross-attention -> score.
(1) and (2) depend only on the user and run offline. Only (3) runs at request time.

(1) Causal Transformer

Each past event (item, action, surface, timestamp) becomes a token and goes through 4 causal Transformer layers, up to 8,192 events over 180 days. The causal mask means appending new events never changes earlier outputs, so training encodes a user's whole sequence once and every request reads its own cutoff from it.

(2) Compression: recent 32 + older 32

Thousands of outputs are too many to attend over. The last 32 are kept; older ones are split into blocks and attention-pooled to at most 32 vectors. 64 in total, regardless of history length.

This does not mean only recent behavior counts: the encoder already mixed the full history into those last 32 positions. How far back the model sees is set by the encoder; how much each candidate reads is fixed at 64. History can keep growing offline while online reads stay flat.

(3) Candidate target attention

Candidates use the same tokenizer and embeddings as history events, with the action field set to "unknown." The candidate is the query; the 64 vectors are keys and values; two layers.

Two kinds of sharing keep it cheap. About 1,000 candidates per request read the same KV. Same idea as HSTU, except HSTU recomputes the user side on every request, while here the KV is precomputed and shared across requests. And grouped-query attention lets 32 heads share 4 KV sets, cutting stored KV to 1/8.

The output joins the usual features in a DCNv2 + Deep&Wide head.

Results

A/B against the production HSTU-style model, 20% traffic each, 7+ days: CTR +6.02%, conversion +4.45%, DAU +0.26%. Gains held across heavy, medium and light users and over a five-week holdout.

Takeaways

- Precomputing the user side is old. What's new is the compression layer. Store one vector and candidates can't pick relevant behavior; cache the full sequence KV (as in OneTrans) and storage grows with history; 64 compressed vectors get both.
- Very close to Kuaishou's UxSID: both compress an ultra-long sequence into a few vectors offline for candidates to read. They differ in how they compress (by interest vs by time), where candidate interaction happens (offline via Semantic IDs vs online with full features), and whether the sequence is encoded first.

#RecommenderSystems #LinkedIn #MachineLearning #RecSys #AI
