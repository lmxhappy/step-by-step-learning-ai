# YouTube Music new-song listening up 11%: industrial-grade zero-shot cross-domain knowledge distillation

> [中文版](youtube-music-cross-domain.md)

### Paper: Zero-shot Cross-domain Knowledge Distillation: A Case study on YouTube Music
### Link: https://arxiv.org/pdf/2603.28994
### Company: Google YouTube team
### Idea: transfer learning
### Area: cross-domain learning + transfer learning

## Overview:
The paper proposes a zero-shot cross-domain knowledge transfer method: "distill" the knowledge of YouTube's very large video recommendation teacher model into the YouTube Music recommendation model, which has only 1/100 of the traffic. This greatly improves the ranking model in a low-traffic scenario while keeping inference latency very low. They **completely dropped the sampled video data they had long relied on**, so training is more stable and iteration is faster.

Put simply, the teacher model generates soft labels that serve as a new label for the student model to predict, which improves the model. Specifically:
### (1) Getting soft labels
Offline, the teacher model (the YouTube model) generates soft labels for YouTube Music samples. Because the features differ, missing features are filled with global mean statistics or 0.

### (2) Modifying the student model
Where the student and teacher share an objective, such as CTR and watch/listen time, the output layer of the relevant task tower goes from 1 output to 2: the first is the original output, the second predicts the teacher's soft label. The loss is KL or MSE, with the same weight as the main task, trained jointly.

YouTube Music has two main scenarios: homepage recommendation and radio.
* Homepage: like YouTube, it has CTR and watch-time tasks, so soft labels are used for both to improve training. YouTube has no matching task for Discovery, so the Discovery task is not distilled.
* Radio: YouTube has no matching task, so a task YouTube does have, "continue watching", is introduced with an added tower. It can be trained on music data alone and is not used at inference.

### **A/B**: compared with a model trained on YouTube Music data only
Homepage: engagement +0.58%, discovery +1.12%, new-song listening +11.39%
Radio: engagement +0.70%, discovery +2.13%, new-song listening +0.96%

## Takeaways:
* Many people thought of this idea long ago, and parts of it have long been in use, but getting this "plain" method to actually work across such a large cross-domain gap, launch it stably, and publicly share production data makes this paper the first public industrial case so far. The method really is as plain as "lend the big model's brain to the small model for a while", but for the combination of cross-domain + zero-shot + production launch, it still counts as a "plain but effective" industrial innovation rather than a routine thing "everyone already does".
* Before this, YouTube Music used cross-domain data augmentation, i.e. sampling YouTube data and stuffing it into the music model's training, which made training unstable and created a data dependency.
* The method has general value for short video, news, e-commerce, and indeed any low-traffic recommendation scenario: knowledge from a big domain can finally be "lent" cleanly to a small one.

## Caveats
The paper calls the modified task towers "Pre-existing Task Layer(s)", which is easy to misread: it's tempting to think these are the original task towers with no changes at all.

## Credibility: production

## Recommendation: practical value

![pic](../imgs/youtube-music-cross-domain.png)
