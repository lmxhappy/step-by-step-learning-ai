# Recommendation algorithm papers

> [中文版](2025-01-30-kuaishou-diffusion.md)

```
1 Recommendation: for now there are only two levels: "worth a look" and "read if you have time". Occasionally "strongly recommended".
2 * Worth a look: people working on ranking should get to know it soon; it helps with work.
3 * Read if you have time: as the name says.
4
5 Credibility: offline or production.
6 * Production: methods that can be used in production; relatively more useful, and others are more willing to use them in their own business.
7 * Offline: by comparison, less so.
```

1 Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation

Link: https://arxiv.org/pdf/2501.17670

Company: Kuaishou

Area: Vector Quantization (real-time clustering) + behavior sequences + Diffusion

Overview: uses two techniques, Vector Quantization (real-time clustering) and Diffusion, to improve user interest modeling from behavior sequences. (1) Vector Quantization: real-time clustering maintains a codebook that stores the cluster centroids. (2) Diffusion: the forward process of the diffusion model gradually adds noise to the true next-item representation, and the reverse process gradually recovers it from noise, guided by the user's interaction sequence; the goal of denoising is to get an accurate next-item representation.

I can't help marveling at how powerful k-means is: yet another paper built on it. Clustering improving results has been shown to work by several companies on behavior sequences, user representations, item representations and more. Dear readers, consider applying it to your own scenarios too, and ideally publish a paper about it.

I'm curious why the author list doesn't include the big names Gai Kun and Zhou Guorui.

Credibility: offline

Recommendation: worth a look
