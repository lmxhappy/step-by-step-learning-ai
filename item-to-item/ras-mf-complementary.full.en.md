# Two matrix factorizations for complementary recommendation: factorize user–item and filter by category, or factorize the co-purchase matrix directly

> [中文版](ras-mf-complementary.md)

### Paper: Comparison of Matrix Factorization Methods for Item-Based Recommendations (Journal of Computer and Systems Sciences International, 2025)
### Link: https://doi.org/10.1134/S1064230725700686
### Company: Federal Research Center "Computer Science and Control" of the Russian Academy of Sciences (data from an unnamed e-commerce platform, about 13M products)
### Idea: matrix factorization, co-occurrence statistics, rule-based filtering
### Area: i2i (complementary product recommendation)

## Overview:
The paper compares two ways of doing complementary recommendation with matrix factorization, both using LightFM.

- **Approach 1:** factorize the user–item purchase matrix as usual and find neighbors with the item vectors. Item vectors find similar products, so at retrieval time category co-occurrence statistics are used to mark out "likely complementary categories", and the search is restricted to those.
- **Approach 2:** factorize the item–item co-purchase matrix directly, so the complementary relation goes into the vectors at training time.

Evaluation takes the last item added to a basket as the answer and the earlier items as triggers. Offline metrics are split between the two; in an online A/B test, approach 2's revenue lift was more than twice approach 1's, but the paper gives no concrete numbers.

### Data and evaluation
**Data.** Purchase records from an e-commerce platform; products carry gender, age group, category, brand, seller, text description, image and price. Users and products with too few purchases are removed first, with a threshold between 5 and 10 depending on data volume.

**Evaluation.** The last item a user added to the basket and bought is the target, the earlier items serve as triggers, and the question is whether the model's top N contains it. Metrics are precision@k, MRR@k and Gini@k; a lower Gini means more spread-out recommendations.

### Model side
#### Approach 1: factorize user–item, filter by category at retrieval
**Training.** ALS, BPR and LightFM (with the WARP ranking loss) were compared on the user–item purchase matrix. LightFM was best and is used from then on. LightFM can take content features such as product attributes, but the paper only says it was trained on purchase interactions and doesn't mention any content features, so the LightFM here is effectively matrix factorization trained with the WARP loss. Vectors are 100-d, trained for 50 epochs, with up to 15 negatives sampled per positive. Interaction weights are purchase counts, normalized with BM25.

**Retrieval.** Given a trigger, neighbors are found by cosine similarity of item vectors. Searching directly mostly returns other versions of the same product, so the search is restricted to "trusted complementary categories", defined as follows:
- Rules: categories must differ; gender and age group must match; price must not exceed the trigger's (a case for a phone, not the other way round); brand, seller, text and image are not used.
- Statistics: for each category, compute PMI of its co-occurrence with other categories within sessions, and keep only categories whose PMI is above an upper quantile.

#### Approach 2: factorize the item–item co-purchase matrix directly
**Building the matrix.** Two products bought by the same user within a time window count as one co-occurrence, giving an item–item co-occurrence matrix. Windows from 3 to 20 days were tried, and 10 days was best. Product pairs with too few co-purchases are dropped.

**Model.** LightFM is carried over from approach 1; ALS and BPR were not tried again on the co-purchase matrix. After filtering out infrequent product pairs, training became over 20x faster, so the model was enlarged: 384-d vectors, 200 epochs, up to 50 negatives sampled per positive.

**Factorization.** The normalized co-occurrence matrix is factorized into two sets of vectors, "previous item" and "next item", because complementarity has a direction: phone first, case later. Recommendation searches for neighbors of the trigger's vector; using the "next item" vectors works best, better than the "previous item" set or crossing the two. BM25 weighting helps here too.

**Where to filter.** At first approach 1's rules are applied when building the matrix: the recommendations look accurate, but coverage is too narrow and metrics are low. The rules are then moved step by step from matrix building to retrieval: first only "categories must differ" is kept when building the matrix, and finally no filtering is applied when building it, apart from dropping product pairs with fewer than 15 co-purchases. Product coverage grows from 3M to 15M, and the metrics go up with it.

### Results
**Online:** an A/B test on the complementary recommendation slot found that approach 2's revenue lift was more than twice approach 1's.

## Takeaways:
* If you want complements rather than similar items, turning co-purchases directly into training data beats filtering user–item vectors after the fact: the complementary relation goes into the vectors.
* Complementarity has a direction, so "previous" and "next" items can be learned as two separate sets of vectors; in this paper, retrieving with the "next" set worked best.
* Rule filtering works better at retrieval than when building training data: keeping more training data raises both coverage and metrics.

## My take
* The online A/B test is a single sentence ("more than twice the revenue lift"), with no baseline, absolute values, traffic or duration.
* Evaluation treats the last item in the basket as the answer, but the next item isn't necessarily a complement of the earlier ones. The trigger is several earlier items, and the paper doesn't say how recommendations for multiple items are merged.
* The evidence for "10 days is best" is thin: in Table 5, precision@10 is 0.00767 for 5 days and 0.00766 for 10 days, essentially the same.
* The conclusion says approach 2 is "more accurate beyond the top ten", but the tables only go up to @10. The paper doesn't say what time span approach 2 used, nor whether the two approaches were compared on the same data.
* The method combines ALS, BPR, LightFM and co-occurrence statistics; the paper's contribution is mainly the experimental comparison and where the filtering rules are placed.

## Credibility: production

## Recommendation level: worth a look if you have time
