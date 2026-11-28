# Outfit matching: a GBDT extracts cross-item rules from the attributes of two items, each rule is encoded from its attribute crosses and scored together with image vectors; attention weights and per-cross scores explain "why they match"

> [中文版](nus-aic.md)

### Paper: Interpretable Fashion Matching with Rich Attributes (SIGIR 2019)
### Link: https://doi.org/10.1145/3331184.3331242
### Institution: National University of Singapore, University of Science and Technology of China, Hefei University of Technology
### Ideas: GBDT decision paths as rules, attribute-cross rule embedding, attentive rule reweighting, visual-rule joint scoring, BPR, interpretable complementary matching
### Area: i2i (complementary recommendation, outfit matching, explainable recommendation)

## Breakdown:
When a user is looking at a piece of clothing (e.g., a top), recommend matching items from complementary categories (bottoms, shoes) and say why they match. The paper first trains a GBDT whose input is the attributes of the two items concatenated; each tree's root-to-leaf path is a cross-item rule (e.g., "top material = wool AND bottom material = wool AND top category = blazer"). Each rule is encoded into a vector from the crosses of adjacent attributes on it, an attention network weights the rules, and three scores computed together with the two items' image vectors are summed.

Existing outfit matching relies mainly on images and pulls matching items together in a vector space; it is a black box that can't say why two items match. Items come with rich attributes (category, color, material, pattern, neckline, style) that are largely unused, and hand-written matching rules depend on domain knowledge and have limited coverage.

![pic](../imgs/nus-aic.png)

### Method
An item pair's score is the sum of three terms: image matching, rule matching, and the interaction of the two. Rule matching uses the pair's T rule vectors (one per tree), encoded from the rules a GBDT extracts. The GBDT is trained first and frozen; everything else is trained together with BPR.

#### Scoring: image, rule and interaction terms
The three terms below are summed as the pair's total score. Each term is a 64-dim vector, which takes an inner product with its own learnable weight vector before the sum.
* **Image matching.** The trigger item and the target item both go through a pretrained ResNet-50 to extract pool5 features, and one linear layer compresses them to 64 dims. Both items share this image encoder. Image matching is the element-wise product of the two image vectors.
* **Rule matching.** Using weights from a gating network, the T rule vectors (one per tree; how they are built is in the next section) are averaged into the pair's rule vector.
    * **Attention network (gating network).** For a given rule, multiply (query item's image vector + rule vector) element-wise with the candidate's image vector, concatenate the result with the rule vector itself, pass it through one ReLU hidden layer, then take an inner product with a weight vector to get a scalar score; a softmax over the T rules' scores gives each rule's weight.
* **Image-rule interaction.** The sum of the two image vectors multiplied element-wise with the same rule vector.

**Loss.** BPR, requiring positive pairs to score higher than negative pairs. Positives are two items that appear in the same outfit. Each positive gets 3 negatives, sampled at random from items in a category different from the query item's, which never appeared in the same outfit as the query item.

**How explanations are given.** Two levels. At the rule level, take the rules with the largest gating weights. At the attribute-cross level, use the fact that the rule-matching term is linear (a rule vector is the average of its cross vectors): each cross's contribution to the score is "rule weight × inner product of the rule-matching term's weight vector with that cross vector", and the largest one is picked. For example, for a navy coat with low-rise gray jeans, the top contributor in the first rule is "bottom low-rise & top style sophisticated"; for a white sleeveless cutout dress with white heels, it is "dress white & shoes white". Explanations come only from the rule-matching term; the image-matching and interaction terms are not part of the explanation.

**Why the gating network's input is asymmetric.** The query item and the candidate take different positions in the input, matching directed pairings such as "top to bottom".

**Why add the interaction term.** It lets image and rules interact in the same space, effectively using attribute rules to correct image matching. On women's, using only the interaction term (MRR 33.12) is on par with using all three (33.18).

#### Where the T rule vectors come from
Each tree gives the pair one rule, and each rule is then encoded into a vector, in two steps.

##### (1) GBDT rule extraction
* **Input.** The two items' attributes are each one-hot encoded and concatenated as "the top's attributes, the bottom's attributes".
* **T rules per pair.** Trained with XGBoost (10 trees, max depth 6 by default). An item pair lands in one leaf of each tree, and the root-to-leaf path is recorded as a rule, i.e., a sequence of "attribute + yes/no". Each tree gives one, so T is the number of trees, 10 per pair by default. Attributes of both items can alternate along a path, so a rule is a cross-item, high-order attribute cross. A column appears at most once on a path, but the same attribute can appear several times (e.g., "top material = wool" answered no, then "top material = cotton" answered yes; or the material of the top and of the bottom each asked once). Only the paths are used, not the leaf predictions; the trees are frozen after training and not updated afterwards. The paper doesn't say what samples or labels the trees are trained on.

##### (2) Encoding each tree into a vector
For this item pair, each tree is encoded into a 64-dim vector.
* **Attribute + decision.** Each attribute has a 64-dim vector, and "yes" and "no" each have one shared vector; the attribute vector plus its decision vector represents "top material = wool, decision yes". Only (number of attributes + 2) vectors are needed. The paper doesn't make clear whether the same value on the two items shares one attribute vector. A rule thus becomes a sequence of vectors, one per node on the path, at most 6.
* **Attribute crosses.** For windows of 2, 3, … adjacent attributes on the path, take the element-wise product; each window gives one cross vector. Only contiguous windows on the path are used, not arbitrary combinations. For the deepest path of 6 nodes, there are 5 second-order windows (1,2) (2,3) (3,4) (4,5) (5,6), 4 third-order (1,2,3) (2,3,4) (3,4,5) (4,5,6), 3 fourth-order (1,2,3,4) (2,3,4,5) (3,4,5,6), 2 fifth-order (1,2,3,4,5) (2,3,4,5,6) and 1 sixth-order (1,2,3,4,5,6), 15 cross vectors in total. The paper doesn't say whether the cross order is capped; its case study only shows second-order crosses.
* **Averaging.** The average of all cross vectors is the rule's vector. Each item pair thus gets T 64-dim rule vectors, which go to the rule-matching term above, weighted first by the gating network.

**Why not use the leaf scores.** The authors argue that trees generalize poorly to unseen attribute combinations, since such inputs tend to go "no" all the way down.

**Why build rule vectors from attributes.** The more direct approach treats each leaf as an ID with its own independent vector, ignoring which attributes the rule contains. That way rules share no parameters, and the parameter count grows linearly with the number of leaves. When the number of trees goes from 10 to 100, the ID version's hit@5 on men's drops from 39.25 to 37.88, while the attribute-based version stays flat (39.51, 39.45).

### Inference
* **Pairwise scoring.** Each candidate must be concatenated with the query item's attributes and run through the GBDT to get T paths, then through the gating weights and the three terms; candidate vectors can't be precomputed for vector retrieval.
* **Evaluation setup.** Each positive test pair gets 500 negatives (sampled from all categories other than the query item's), and the 501 candidates are ranked. The paper doesn't discuss retrieval over the full catalog.

### Data and results
The authors crawled 30,790 outfits from the Lookastic outfit website and treat two items in the same outfit as a positive pair. Men's has 5,069 items and 125k pairs; women's has 10,016 items and 159k pairs. Attributes come from the site's annotations, supplemented with Visenze's automatic tagging API: 65 colors, 38 materials, 40 patterns, 253 fine-grained categories, 11 styles and 114 category-specific attributes. Random 70%/20%/10% split, repeated 5 times and averaged. All baselines use the same image features and BPR loss (except the Siamese network). Offline experiments only.
* **Overall.** MRR is 30.74 on men's and 33.19 on women's, versus 29.10 and 31.63 for the strongest baseline TEM (which also uses GBDT paths but one ID vector per rule), relative gains of 5.6% and 4.9%. The image-only BPR-DAE gets 23.35 and 23.69.
* **Three-term ablation (MRR, men's/women's).** Rule term only 18.90/23.40, interaction term only 29.22/33.12, image term + rule term 30.38/32.73, all three 30.74/33.18.
* **Gating network.** Replacing it with average pooling drops hit@5 from 39.51 to 37.70 on men's and from 43.83 to 42.69 on women's; max pooling gives 38.01 and 42.98.

## Takeaways
* GBDT root-to-leaf paths over the concatenated attributes of two items give cross-item attribute rules for free, with no hand-written matching rules.
* Encoding a rule as the average of its attribute crosses lets rules share parameters, which keeps the model stable as the number of trees grows, where per-leaf ID vectors overfit.
* Because the rule term is linear in the cross vectors, each attribute cross's contribution to the score can be read off directly.

## Caveats
* The explanations cover only the rule term, which on its own is the weakest component (MRR 18.90/23.40 versus 30.74/33.18 for the full model), while the interaction term alone nearly matches the full model on women's; the explanations therefore account for a minor share of the score.
* At the default 10 trees, attribute-based and ID-based rule vectors differ by only 0.26 hit@5 on men's (39.51 vs 39.25), while TEM gets 37.88, so at the default setting most of the gain over TEM comes from the joint scoring terms; the attribute-based encoding mainly matters when the number of trees is large.
* Test negatives are drawn from every category except the query's, so many of them are not in the positive's category, and category compatibility alone can rank part of them; the split unit (pairs or outfits) is not stated.
* No code is released, and several details are unspecified (the GBDT's training samples and labels, the maximum cross order, whether the two items share attribute vectors); there are also small numeric slips (BPR-DAE's hit@10 of 30.90 on men's is below its hit@5 of 30.97; women's MRR is 33.19 in Table 1 and 33.18 in Table 3).

## Credibility: offline

## Recommendation level: worth a look if you have time
