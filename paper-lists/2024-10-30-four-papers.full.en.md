# Recommendation paper list (2024.10.30)

> [中文版](2024-10-30-four-papers.md)

> Source: WeChat post https://mp.weixin.qq.com/s/TGlyp1svAwicQpRdMqJOfw, published 2024-10-30.

1. SimRec: Mitigating the Cold-Start Problem in Sequential Recommendation by Integrating Item Similarity

a. Link: https://arxiv.org/pdf/2410.22136

b. Company: Amazon

c. Direction: behavior sequence modeling + item cold start + contrastive learning. Defines a new loss that incorporates contrastive learning, enabling good recommendations for new items. Skimmed it: the method is very simple; details still need closer study.

d. Rating: strongly suggested

2. Deep Group Interest Network on Full Lifelong User Behaviors for CTR Prediction

a. Link: https://arxiv.org/pdf/2311.10764

b. Company: Meituan

c. Direction: long behavior sequences. Considers all of a user's behaviors and models multiple interaction types. Also uses group embeddings to boost the model, a recently popular enhancement trick. On top of that, it gives a data analysis that proves the necessity of the approach, which is even more valuable: very instructive for how we find problems and design solutions.

d. Rating: strongly recommended

3. Enhancing CTR prediction in Recommendation Domain with Search Query Representation

a. Link: https://arxiv.org/pdf/2410.21487

b. Company: Huawei

c. Direction: cross-domain + contrastive learning. Data from the search scenario helps improve the recommendation scenario.

d. Rating: worth reading

4. Modeling Temporal Positive and Negative Excitation for Sequential Recommendation

a. Link: https://arxiv.org/pdf/2410.22013

b. Company: an Australian company I don't know

c. Direction: behavior sequence modeling. Typical behavior sequence modeling aims to capture users' dynamic interests, ignoring static interests and only using items with positive behaviors. This paper considers both dynamic and static interests, and also items with negative behaviors.

d. Rating: read if you have time. Don't expect production gains; I just found it interesting. If you have time, it broadens your horizons.
