# Tops to bottoms: hand-screen 15 matching rules, let attention weight the rules per sample, and distill them into an image-text scoring network

> [中文版](sdu-akd.md)

### Paper: Neural Compatibility Modeling with Attentive Knowledge Distillation (SIGIR 2018)
### Link: https://arxiv.org/abs/1805.00313
### Institution: Shandong University, National University of Singapore, Tencent AI Lab
### Ideas: knowledge distillation, logic-rule teacher, attention over rules, BPR, image-text item encoder, complementary clothing matching
### Area: i2i (complementary recommendation, outfit matching)

## Breakdown:
Given a top, rank candidate bottoms by how well they match (tops and bottoms that fashion experts put together on Polyvore are the positives). The paper first mines 15 matching rules from the training data (e.g., "coats go with dresses", "silk tops shouldn't go with knit bottoms"), then trains an image-text scoring network (the student). On every training sample, the rules it triggers rewrite the student's scores to form a teacher, an attention network sets each rule's weight per sample, and the student learns both from the real matches and from the teacher.

Existing matching models are purely data-driven and ignore the common sense accumulated in fashion. Rules, on the other hand, are fuzzy: the same rule (e.g., "stripes go with stripes") fits some samples and not others, so giving it one weight for all samples is a poor fit.

![pic](../imgs/sdu-akd.png)

### Method
Rules are mined offline first ((1)). During training, each batch does three things in order: the student scores ((2)), the rules rewrite the student's scores into a teacher ((3)), and the student learns from both the ground truth and the teacher ((4)).

#### (1) Mining matching rules
* **Attribute values.** The authors take five attributes as the main drivers of matching (color, material, pattern, category, brand) and collect each attribute's values from the titles and categories in the training data (e.g., black, white, green for color; knit, silk, leather for material).
* **Candidate rules.** Over the positive pairs in the training set, count how often each "top value + bottom value" combination co-occurs per attribute, and keep the 10 most frequent pairs (candidate positive rules, e.g., "black top with black bottom") and the 10 least frequent pairs (candidate negative rules, e.g., "blouses shouldn't go with dresses") for each attribute.
* **Manual screening.** Three fashion enthusiasts screen them by hand, leaving 15 rules (9 positive, 6 negative).
* **Deciding whether a pair triggers a rule.** If the rule's first value appears in the top's text (title and categories) and the second value appears in the bottom's text, the pair triggers the rule. It is plain string matching.

#### (2) Student network: encode tops and bottoms separately, score by inner product
The student is the network that does the final scoring. Each item has two inputs (the image goes through an ImageNet-pretrained AlexNet and the 4096-dim fc7 output is taken; the title and category text goes through a TextCNN over word2vec word vectors, giving 400 dims). Vision and text are not scored separately: the two are concatenated and passed through one fully connected layer (1024 dims, sigmoid) to get the item's vector. Tops and bottoms each have their own fully connected weights, not shared; in the code, the TextCNN is shared by tops and bottoms and trained with the model (word vectors frozen), while image features are extracted in advance and not trained. The match score m_ij between top i and bottom j is the inner product of their vectors.

**Loss.** A BPR pairwise loss. Each sample is a triplet "top i, a positive bottom j, a random bottom k", and m_ij should be higher than m_ik; k is drawn at random from bottoms never paired with i, 3 per positive pair.

#### (3) Teacher network: rewrite the student's scores with the rules
The teacher has no encoder of its own; it just adjusts the student's scores on the triplet according to the rules.
* **Whether a rule applies.** A rule applies only when it can tell j and k apart (for a positive rule, if exactly one of j and k triggers it with the top, that one is rewarded; for a negative rule, if exactly one triggers it, the other one is rewarded). If both or neither trigger it, the rule does not apply.
* **Rewriting the scores.** The student's two scores [m_ij, m_ik] are first normalized into a distribution that sums to 1; for each applicable rule, the rewarded side is multiplied by exp(C × rule weight), and the result is renormalized to give the teacher's distribution. For example, if the top is a coat, j is a dress and k is shorts, "coat + dress" fires for j, and the teacher leans toward j more than the student does.
* **Rule weights (attention).** Several rules can apply to the same triplet. For each rule, the representations of the top, the positive bottom and the negative bottom, plus the rule's one-hot code, are each multiplied by a matrix and summed (the two bottoms share one matrix), passed through a nonlinearity, and dotted with a vector to get a score; a softmax over the rules that apply to this triplet gives the rule weights. In the paper the three items are represented by their raw image-text features; in the code, by the student's 1024-dim vectors. The attention network is trained together with the student using the loss in (4).

**Why weight rules per sample.** Rules are triggered only by whether a word appears in the text, which is coarse. Two samples can both trigger "stripes + stripes": in one the stripes on top and bottom really echo each other, in the other the stripe width and colors clash, and the rule shouldn't decide the latter. Attention sees the items themselves and can lower the weight in the second case.

#### (4) Distillation: the student learns from the ground truth and the teacher
**Loss.** The total loss is (1 − ρ) × BPR loss + ρ × cross-entropy (teacher distribution as the target, student distribution as the prediction) + L2 regularization. ρ starts at 0 and grows during training (in the code, 1 − 0.95 to the power of the epoch).

**Why ρ grows over time.** The teacher is derived from the student; early on the student hasn't learned much, so the teacher isn't reliable either. Training therefore fits the ground truth first and gradually shifts weight toward learning from the teacher.

**Code vs. paper.** The official code (model/matching_attention_classes.py and model/matching_attention.py, in Theano) differs from the paper in a few places. The attention input is the student's item vectors rather than the raw image-text features; the coefficient C is applied before the softmax (acting as a temperature) and not again when rewriting the scores; training runs for 25 epochs (40 in the paper) and the fully connected layer uses a hard sigmoid; and the released script configures only 3 rules.

### Inference
* **Student scoring (AKD-DBPR-p).** Compute top and bottom vectors separately; their inner product is the match score. Rules and the teacher are not needed, and bottom vectors can be precomputed.
* **Teacher scoring (AKD-DBPR-q).** The rules can also rewrite the student's scores on test triplets, but this needs to know which rules each of the two bottoms triggers, so it can only compare a pair of bottoms.
* **Retrieval experiment.** Each test top is paired with T random bottoms (exactly 1 positive), they are ranked by the student's inner product, and the rank of the positive is measured.

### Data and results
FashionVC (20,726 expert outfits from Polyvore, 14,871 tops and 13,663 bottoms). Positive pairs are split 8:1:1 into train, validation and test, and each positive pair gets 3 random negatives. The main metric is AUC over triplets (the fraction where the positive outscores the negative). Offline experiments only.
* **Overall.** Student AUC 0.7843, teacher 0.7852; the same architecture without the teacher (DBPR) gets 0.7704, and the strongest external baseline, BPR-DAE (the group's earlier image-text autoencoder method), 0.7616.
* **Attention.** With uniform rule weights, the student gets 0.7785 and the teacher 0.7760.
* **Single modality.** Image only: DBPR 0.6826, and with rules the teacher gets 0.7302 and the student 0.6961. Text only: DBPR 0.7181, teacher 0.7374, student 0.7345.
* **Single rules.** Using one rule at a time, student AUC ranges from 0.7714 to 0.7814; negative rules and category rules generally do better, and the best is "no blouses + dress" (0.7814).
* **Retrieval.** 64.59% of the test tops never appear in the training set. MRR is only shown as curves: with 10 candidates the student is about 0.50 versus about 0.48 for DBPR, and with 50 candidates both are about 0.22.

The code is open source.

## Takeaways
* Turning a handful of rules into a teacher that reweights the student's own pairwise scores injects domain knowledge without changing the model that serves.
* Negative rules ("X shouldn't go with Y") and category rules helped more than color or material rules.
* Per-sample rule weights beat uniform weights, but the total gain over the no-rule student is about 1.4 AUC points.

## Caveats
* The released training script keeps the epoch and hyperparameter setting with the best test-set accuracy rather than the best validation accuracy, while the paper says it reports test performance after a fixed 40 epochs; no significance tests are given for gaps of around 0.01 AUC.
* A single rule ("no blouses + dress", 0.7814) already brings the student close to the full attentive version with 15 rules (0.7843) and above the uniform-weight version with all rules (0.7785), so the separate contribution of the attention over many rules is small.
* In the code, gradients flow through the teacher distribution into the attention network, so the distillation term also pushes the rule weights toward whichever bottom the student already prefers; the learned rule confidences therefore partly reflect agreement with the student.
* The paper and the code differ in the attention input (raw features vs. student vectors), where C is applied, the number of epochs (40 vs. 25), and the activation, and the released script configures 3 of the 15 rules.

## Credibility: offline

## Recommendation level: worth a look if you have time
