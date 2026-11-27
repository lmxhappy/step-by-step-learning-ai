# Outfit matching: automatically crop a million outfits out of on-site collage images to train style embeddings; online, predict the category, expand it to complementary categories, then run ANN within each category

> [中文版](pinterest-ctl.md)

### Paper: Bootstrapping Complete The Look at Pinterest (KDD 2020)
### Link: https://arxiv.org/abs/2006.10792
### Institution: Pinterest
### Ideas: complementary fashion recommendation, outfit dataset mining, object detection, style embedding, triplet loss, category-restricted negative sampling, category expansion, per-category ANN retrieval
### Area: i2i (complementary recommendation, outfit matching)

## Breakdown:
When a user views an apparel product on Pinterest (e.g., a red striped blouse), recommend products from other categories that complete the outfit (shoes, sunglasses, jewelry). The paper first crops 1 million outfits automatically out of on-site outfit collage images, then uses them to train an image model (the paper calls it the CTL model; CTL abbreviates Pinterest's outfit feature Complete The Look and is not a method name) that outputs both the product's category and a 128-dim style embedding. Online, it predicts the query product's category, expands it into several complementary categories, and runs approximate nearest neighbor (ANN) search on style embeddings within each category.

Existing outfit systems either rely on products hand-picked and matched by stylists, or on past user engagement ("people who bought this also bought…"), and a newly launched product has neither. Public Polyvore outfit datasets have only 20k to 70k outfits, and the site has shut down. Academic outfit models are mostly classifiers that score a whole outfit, not designed for retrieval over tens of millions of products.

![pic](../imgs/pinterest-ctl.png)

### Method
Two steps: first generate an outfit dataset automatically ((1)), then train the image model on it ((2)).

#### (1) Mining outfits from Pinterest images
The goal is a large number of "items from the same outfit" as positives, with item crops that look like the white-background product images served online.
* **Filter collages.** An existing on-site image style classifier (input is Pinterest's general-purpose image embedding, Unified Embedding; output is a score for each class such as collage, product shot on white background, stock photo, full outfit and cropped outfit) picks out Polyvore-style collages (several items laid out on one white-background image) at a threshold of 0.9, yielding about 35M images. Collages are chosen because the cropped items have clean backgrounds, close to product images.
* **Detect items.** The on-site fashion detector (Faster R-CNN trained on 251k images) boxes every item and its category, keeping only 21 apparel categories; about 10M images have at least one detected item. The boxes in one image form one outfit.
* **Clean up.** Drop overlapping boxes and boxes smaller than 5% of the image; require 3 to 8 items and at least 3 different categories per image; drop outfits where all items share one color, so the model doesn't just learn to match colors. Filtering by clicks, repins or image age was also tried, with no visible difference.

The result is 1,006,519 outfits and 4.246M items, with 24,960 outfits held out for testing.

#### (2) Image model: category + style embedding
One image model outputs both the category and the style embedding; query products and catalog products are encoded by the same model, with no separate towers.
* **Backbone.** SE-ResNeXt101, initialized from the weights of Unified Embedding (Pinterest's multi-task visual model shared by visual search products such as Lens and Shop The Look); its last-layer features feed two heads.
* **Category head.** Two fully connected layers predicting one of 21 categories, with labels from the detector in (1); accuracy 98.15%. Online it decides the query's category and is used to build per-category catalog indices.
* **Style head.** Two fully connected layers (with BatchNorm and dropout 0.5) output a 128-dim style embedding. Whether two products go together is just the Euclidean distance between their style embeddings; there is no separate scoring network. The paper doesn't say whether the two heads are trained jointly or separately.

**Loss.** Three training methods are compared; production uses the triplet one.
* **Triplet loss (in production).** Two items of different categories from the same outfit are the anchor and positive; the negative is a random item from another outfit in the same category as the positive (if the anchor is a top and the positive is the shoes from that outfit, the negative is a pair of shoes from another outfit). The squared anchor-positive distance must be smaller than the anchor-negative one by at least a margin. "Anchor and positive in different categories" is not in the paper's text; this follows the sampling script in the official dataset repo.
* **Contrastive loss.** Positive pairs are restricted to two items of different categories from the same outfit; negative pairs are random regardless of category, at a 1:1 positive-to-negative ratio.
* **Proxy classification loss.** Each outfit is a class, and an item image must be classified into its own outfit; each softmax is over only 2048 sampled outfits (following how Unified Embedding is trained).

**Why negatives are sampled by category.** When the negative is in the same category as the positive, the model can't tell them apart by category differences like "is it a shoe"; it must learn "does this shoe go with this top". Forcing positive pairs in the contrastive loss to differ in category likewise keeps the model from treating "looks alike" as "goes together".

### Inference
* **Indexing.** Every catalog product goes through this model offline to get its category and style embedding; an inverted index is built per category, and each category gets its own ANN index. Only products the image style classifier labels as white-background product shots enter the index, and queries likewise trigger only on product shots; stock photos and outfit images are excluded.
* **Category expansion.** When a user views a product, the model predicts its category and style embedding in real time, and the category is expanded into several complementary categories (e.g., "Shirts & Tops" expands to shoes, sunglasses, and jewelry & watches). The paper doesn't say how the expansion rules are defined.
* **Per-category retrieval.** The query's style embedding searches each complementary category's ANN index for nearest neighbors, giving one list per category.
* **Blending.** The per-category lists are merged into a final order; the paper doesn't say how.

The model is exported to the C++ serving stack via TorchScript.

### Data and results
Offline evaluation uses test outfits with exactly 5 items. Each item is a query against a retrieval pool of 200 items (the other items of the same outfit as positives, the rest randomly sampled, with pools built per category and averaged), measuring Recall@1/5/10. There is also a fill-in-the-blank task (remove one item from an outfit and pick it among the true item and 3 random items of the same category).
* **Training methods.** Triplet loss with category-restricted negatives reaches R@1 of 20.3%, triplet with random negatives 18.5%, contrastive loss 16.0%, proxy classification 15.8%, and Pinterest's earlier "scene image to product" model (Kang et al., 2018, also called CTL) used as is only 3.9%; fill-in-the-blank accuracy is 78.5%, 77.6%, 74.3%, 71.9% and 40.1% respectively.
* **Initialization.** Initializing the backbone from Unified Embedding instead of ImageNet raises R@1 from 17.6% to 20.3%.
* **Data size.** With 10k, 100k and 1M training outfits, R@1 is 12.1%, 16.4% and 20.3%, still rising.
* **Human evaluation.** 120 Pinterest products are run through the full online pipeline, and in-house fashion specialists judge whether each (query, recommendation) pair is compatible. Precision is 55.0% for triplet, 50.6% for proxy classification and 47.7% for contrastive. The specialists report that the model relies too much on color and pattern matching.
* **User study.** As of early 2020 the prototype had only been released internally; 5 users each took part in a one-hour session, and the conclusion was that users see value in the feature. There is no online A/B test or business metric.

The dataset is partly open source (100k training outfits and the full test set, with a triplet sampling script); the model code is not released.

## Takeaways
* Collage images plus an existing detector turn on-site content into a million-outfit training set without any user engagement data.
* Sampling the negative from the positive's category makes the embedding learn compatibility rather than category, worth about 2 points of R@1 over random negatives.
* Serving with one shared encoder, a category classifier and per-category ANN indices keeps complementary retrieval as cheap as visual search.

## Caveats
* There are no online metrics: the system had only an internal prototype, and the user study involved 5 people, so the production value is unmeasured.
* The human evaluation covers 120 query products with no inter-annotator agreement or confidence intervals, and the 55.0% vs. 50.6% gap is reported without a significance test.
* Offline positives come from the same collage images the training data is mined from, and the retrieval pool is only 200 items, while serving runs on product shots over millions of products; the domain gap between crops and product shots is not quantified.
* The complementary-category expansion rules and the cross-category blending are not described, and the paper's text ("any two items from the same outfit") differs from the released sampling script, which also requires the anchor and positive to be in different categories.

## Credibility: offline

## Recommendation level: worth a look if you have time
