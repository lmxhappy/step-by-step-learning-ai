# Outfit completion: encode the existing outfit with a GRU and the candidate with a feed-forward network, multiply element-wise for semantic matching, add a TF-IDF lexical match, and combine both with one linear layer

> [中文版](walmart-ncr.md)

### Paper: Neural Compatibility Ranking for Text-based Fashion Matching (SIGIR 2019, short paper)
### Link: https://doi.org/10.1145/3331184.3331365
### Institution: Santa Clara University, Walmart Labs
### Ideas: text-only fashion compatibility, GRU outfit encoder, asymmetric two-tower, Hadamard interaction, TF-IDF lexical matching, learned weighted scoring, point-wise ranking
### Area: i2i (complementary recommendation, outfit matching)

## Breakdown:
Given a few items already put together (clothing, shoes, a bag), rank candidates by how well each completes the outfit, using text only. Semantic matching: the outfit items go through a GRU, the candidate through a feed-forward network, and the two vectors are multiplied element-wise. Lexical matching: TF-IDF cosine between the outfit's text and the candidate's title. The two are concatenated and passed through one linear layer to get a single score, then a sigmoid gives the compatibility score.

Most existing outfit-matching models rely mainly on images and don't make full use of text descriptions, yet rare words in the text such as brand, material or designer often directly indicate that two products belong together. Semantic matching alone also favors frequent co-occurrences (jacket with jeans) and misses strong signals from rare words like "both are Zara". In addition, an outfit has a variable number of items, so ordinary text-retrieval models can't directly use the whole outfit as a query.

![pic](../imgs/walmart-ncr.png)

### Method
Semantic matching and lexical matching run in parallel, each producing an interaction signal, and the compatibility scorer combines them into one score.

#### Semantic matching
The outfit and the candidate are each encoded into a 32-dim vector and multiplied element-wise, giving a 32-dim semantic interaction vector for the compatibility scorer.
* **Product vector (shared by both sides).** A product's description is split into words, each word is looked up in pre-trained GloVe word vectors (300 dimensions), and all word vectors are averaged (the paper calls this CBOW, meaning a bag-of-words average, not word2vec's training objective). The word vectors are frozen (the authors worry about overfitting).
* **Outfit encoding.** The product vectors of the outfit's items are randomly shuffled and fed one by one into a 1-layer GRU (a recurrent network that updates its hidden state after reading each item; 32-dim hidden state), and the final hidden state is the outfit vector.
* **Candidate encoding.** The candidate's product vector goes through a separate 3-layer feed-forward network (300→128→32, ReLU) into the same 32-dim space as the outfit vector. The two sides don't share an encoder.
* **Interaction.** The outfit vector and the candidate vector are multiplied element-wise (Hadamard product). The result is not summed; the scoring layer weights it.

**Why shuffle.** An outfit is really a set with no order. The authors say in a footnote that item order doesn't affect performance (no numbers given), so there's no need to fix an order such as top, bottom, shoes.

**Why the candidate gets its own network.** The authors argue that a whole outfit and a single product have different statistical properties, so one shared mapping is not flexible enough.

#### Lexical matching
The outfit and the candidate are each turned into a TF-IDF vector (term frequency times inverse document frequency, so rarer words get larger weights), and their cosine similarity (the inner product after L2-normalizing each vector) gives a scalar for the compatibility scorer. This path has no learnable parameters.
* **Outfit (as the query).** The descriptions of the outfit items are concatenated into one passage and turned into a TF-IDF vector.
* **Candidate (as the document).** The candidate's title is turned into a TF-IDF vector. The paper says title here but description on the semantic side, without saying whether they are the same field.

**Why this path is needed.** Rare words such as brand, material or designer tend to get averaged away in semantic vectors, while TF-IDF cosine directly rewards "the outfit and the candidate share the same rare word", e.g., several outfit items are Zara and the candidate is also Zara.

#### Compatibility scoring
The 32-dim semantic interaction vector and the 1-dim lexical score are concatenated into 33 dimensions, passed through one linear layer (33 dimensions to a single score), and then a sigmoid to get the compatibility score. The semantic part is equivalent to a per-dimension weighted inner product between the outfit vector and the candidate vector, with learned weights, so the scoring is neither a fixed dot product nor a multi-layer scoring network.

**Loss.** Point-wise training on (outfit, candidate, compatible or not) triplets with binary cross-entropy; there are 5 times as many negatives as positives. The paper doesn't say how training positives are cut from outfits or where negatives are sampled from.

**Why both paths share one linear layer.** Because both paths are scored by the same linear layer, the lexical score also takes part in the gradients and in turn influences how the semantic part is learned; the authors use this to distinguish the model from an ensemble of two separately trained models. Per-dimension weighting also lets useless semantic dimensions be down-weighted.

### Inference
Compute the compatibility score for each candidate and sort by score. The paper ranks only 101 candidates and doesn't discuss full-catalog retrieval.

Working from the formula (the paper doesn't do this), ANN works, with no loss of accuracy. Let o be the outfit vector and c the candidate vector (both 32-dim), w the first 32 of the linear layer's 33 weights and w₃₃ the last one (the bias is a constant and doesn't affect ranking), and u, v the L2-normalized TF-IDF vectors of the outfit and the candidate (so the cosine is u·v). Then

score = sigmoid( Σ_d w_d·o_d·c_d + w₃₃·(u·v) )

The sigmoid is monotonic and doesn't change the ranking, so only the inside matters. The first term equals (w⊙o)·c and the second equals (w₃₃·u)·v, so together they are a single inner product

[w⊙o ; w₃₃·u] · [c ; v]

The semicolon means concatenating two vectors end to end into one long vector; the inner product of two long vectors equals the sum of the inner products of their corresponding parts. The left side depends only on the outfit and is computed at query time; the right side depends only on the candidate (IDF is fixed once computed on the corpus) and can be precomputed and indexed offline. The whole score is therefore maximum inner product search over vectors with a 32-dim dense part and a vocabulary-sized sparse part, which an index supporting hybrid dense-plus-sparse search can serve. Without such an index, each path can retrieve a batch (dense ANN for semantic matching, an inverted index for lexical matching), and the merged batches are reranked by the full score.

### Data and results
Polyvore outfits (complete outfits curated by fashion bloggers): 21,889 outfits, with 17,316, 1,497 and 3,076 for training, validation and test. Outfits with fewer than 3 items and products whose descriptions have fewer than 3 words are removed, and training and test share no products. In evaluation, one item is randomly removed from each test outfit as the correct answer, and 100 items randomly drawn from other test outfits serve as negatives; the 101 items are ranked. Metrics are MRR (the average reciprocal rank of the correct answer) and HR@k (the share of cases where the correct answer is in the top k). Offline experiments only.
* **Against text-only baselines.** NCR reaches MRR 0.2678 and HR@1 0.1547; among text-only baselines (query likelihood, BM25, DSSM, K-NRM), the highest MRR is query likelihood's 0.1762.
* **Against image+text baselines.** SiameseNet, which uses images, gets MRR 0.2559 and HR@1 0.1320, both below NCR; but on HR@5 and HR@10 NCR (0.3626, 0.5137) is slightly below SiameseNet's 0.3667 and BiLSTM+VSE's 0.5224.
* **Splitting the two paths.** Semantic matching alone gets MRR 0.1708 with HR@1 of only 0.0603 but HR@10 of 0.4296; lexical matching alone gets MRR 0.1794 and HR@1 0.1212 but HR@10 of only 0.2610. Combined, all three are clearly higher: the former gives coverage, the latter puts the right item first.

## Takeaways
* For text-only complementary matching, an exact-match TF-IDF feature on rare words (brand, material) complements dense semantic vectors: adding it lifts HR@1 from 0.0603 (semantic matching alone) to 0.1547.
* An asymmetric two-tower (GRU over a variable-size outfit, MLP for a single candidate) handles "set as query" without concatenating all text into one document.
* A single linear layer over the Hadamard product plus one lexical scalar is still just a weighted inner product plus one scalar term, so the scorer stays cheap.

## Caveats
* The "state-of-the-art" claim holds for MRR and HR@1; on HR@5 and HR@10 NCR trails the image+text baselines, and significance over those baselines is marked only for HR@1.
* The 100 negatives are random items from other test outfits, not matched by category, so the task is easier than ranking same-category alternatives, and a shared brand word is a strong cue in this setup; retrieval from the full catalog is not evaluated.
* How training positives are built and where the 5x negatives come from is not described, and no code is released. The semantic side is described as using product descriptions while the lexical side uses candidate titles (a footnote says titles average about 5 words), so it is unclear whether these are different text fields.
* The lexical-only HR@10 in the ablation (0.2610) is identical to query likelihood's HR@10 in the main table, and the paper doesn't explain the relation.

## Credibility: offline

## Recommendation level: worth a look if you have time
