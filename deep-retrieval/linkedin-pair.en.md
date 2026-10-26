# Embedding retrieval for a small LinkedIn surface

> [中文版](linkedin-pair.md)

### Paper: PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn
### Link: https://arxiv.org/abs/2610.30941
### Company: LinkedIn
### Idea: customizing for the scenario brings the gains
### Area: embedding retrieval, LLM embedding

## Overview:
There is no new technique in this paper. It uses the embedding retrieval everyone already uses to recommend other members to LinkedIn Premium members. The model is a two-tower model plus a weighted average over history; negatives are exposed-but-unclicked plus random.
Before: production used LinkedIn's platform-wide general-purpose member embedding, trained jointly on the engagement objectives of many products, and never custom-trained for this scenario. Now: the LLM profile tower is fine-tuned on this scenario's click data.
Also, the previous recommendation logic in this scenario was "similarity"; it is replaced by click preference instead of profile similarity.

### 1. Model (three towers)

![pic](../imgs/linkedin-pair.png)

**User profile tower**: the résumé on the user's profile page (headline, summary, experience, education, skills, etc.) is concatenated into text and passed through a 0.6B LLM embedding model, producing a 1024-dimensional vector: the user profile vector.

**Candidate tower**: the user's profile page and the candidate's profile page are both profile pages. Candidates share the encoder of the user tower above; it is fine-tuned jointly during training and batch-computed offline every day after launch. This gives the **candidate vector**.

**User tower**: take the behavior sequence of the 10 people the user clicked most recently and turn it into one vector, called the intent vector. Each person gets a score = w·(that person's profile vector + positional encoding); softmax over the 10 scores gives the weights, which are used to average the 10 profile vectors. Both w and the positional encoding are learned, with the most recent click at position 1. The weights do not look at the user themself, so when the user clicks someone new, re-weighting once on CPU is enough; no LLM re-run is needed.

**User vector** = norm(user profile vector + γ × intent vector); each vector is normalized first, and γ is learned.

Retrieval uses the cosine similarity between the user vector and candidate vectors. It also uses "nesting-doll" (Matryoshka) training: the loss is computed not only on the full 1024 dimensions but also on the first 200, 256 and 512 dimensions, and the losses are summed, forcing the leading slices of the vector to work on their own. Production can then keep only the first 256 dimensions, saving storage and retrieval time without training a separate smaller model. This scenario uses all 1024 dimensions.

### 2. Training: negatives and loss

In each exposure session, the clicked candidate is the positive. Each positive gets 3 negatives: people **exposed but not clicked** in the same session come first; if there are fewer than 3, the rest are sampled at random from active Premium members.

The loss is the usual multi-negative contrastive loss: cosine similarity between the user vector and candidate vectors (temperature 0.02), pushing the positive's score above the 3 negatives.

### A/B:
Control is the production general-purpose member embedding. CTR +7.5%, Premium retention +2.1%.

## Takeaways:
* Something customized beats something general-purpose; in machine learning this is almost a sure thing.
* In a membership service people really pay for, it is still worth doing more work.
* LinkedIn does not seem to have that many hands either.

## Caveats
* The real increment is the problem definition, "use click preference instead of profile similarity for professional networking", and getting it deployed.
* The paper keeps saying professional relationships are "complementary": a job-seeking engineer wants recruiters, not another engineer. But nothing in the model targets complementarity; it only learns "who the user will click". Whether what it recommends is actually complementary, the paper gives no evidence. I doubt it.
* As a product, this looks a lot like I2I: the module sits on someone's profile page and recommends other people, where the classic approach is "people who viewed this profile also viewed", found by co-occurrence or similarity. But the query here is not the profile being viewed; it is **the user viewing it**, retrieved from their profile and the people they recently clicked. So as a model it is really U2I (user-to-person). It would be worth trying to add the profile being viewed as an input, as I2I recommendation.

## Credibility: production

## Recommendation: worth a look if you have time
