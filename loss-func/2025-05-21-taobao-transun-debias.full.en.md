# Daily pick (2025.5.21): Taobao debiases regression prediction bias

> [中文版](2025-05-21-taobao-transun-debias.md)

1 TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models in Recommender Systems

Link: https://arxiv.org/pdf/2505.13881

Company: Alibaba Taobao

Area: debiasing (retransformation bias)

Overview:

To fix the prediction bias of MSE regression models, the paper adds a debias network to the existing transformed-MSE model.

The real target is y and the final prediction is y', but the prediction network predicts f(x), whose label is the transformed target (e.g. a log). y' is obtained from f(x) by applying the inverse transform (e.g. an exponential).

The method: besides continuing to train f(x), also train a debias network z(x) at the same time, whose target is y/y'. Don't forget that y' is just a transform of f(x). After training, at inference you just use y'*z(x) as the prediction of y. This is because y/y' = z(x); move y' to the right and you get y = y'*z(x). This is what the authors mean when they say the method is theoretically guaranteed to be unbiased.

What is retransformation bias? When a regression model uses a target transform (log, square root, etc.) to handle non-normally distributed data, the predictions, after being mapped back to the original target space by the inverse transform (retransformation), deviate from the true expected value. That is retransformation bias.

Takeaways:

(1) You can still innovate where there seems to be nothing new to do. I see first-principles thinking at work here, and also the authors' hard work.

(2) Where there is bias, debias it with a debias network.

Credibility: production (Taobao homepage)

Recommendation: worth a look
