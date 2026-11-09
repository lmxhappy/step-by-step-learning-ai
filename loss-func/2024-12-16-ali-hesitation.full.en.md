# Recommendation algorithm papers

> [中文版](2024-12-16-ali-hesitation.md)

1 Hesitation and Tolerance in Recommender Systems

Link: https://arxiv.org/pdf/2412.09950

Company: Alibaba

Area: user retention (this categorization is a bit of a stretch, ha; you'll see why after reading the overview).

Overview: In short, it revises the modeling target and the corresponding cross-entropy loss. Through a user survey, the paper found a kind of user behavior it calls "hesitation". For example, a "click" is not necessarily positive: a user clicks an item, looks into it in depth, and then finds they don't like it. At the same time, the user doesn't give negative feedback. That kind of behavior is hesitation. The paper tried two strategies for it: define it either as a negative sample or as a weak positive sample, and modify the cross-entropy loss accordingly, with the hesitation samples forming a separate third term of the loss. Modeled this way, user retention improved.

When reading this paper, treat it as a user research report and you will get a lot out of it. User research analysis like this is very valuable. I hope to see more of it; it is very useful for people working on algorithms and is a big, good complement to internal data analysis. Also, there is an unhealthy notion going around that in the face of big data, user research has little or even no value. This paper shows that is wrong: both user research and big data have value, and they complement each other. Having big data doesn't mean user research is worthless.

Credibility: online

Recommendation: highly recommended
