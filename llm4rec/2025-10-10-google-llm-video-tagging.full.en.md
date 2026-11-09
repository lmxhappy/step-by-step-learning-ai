# Recommendation Algorithms (2025.10.10): Google's industrial solution for tagging short videos with LLMs

> [中文版](2025-10-10-google-llm-video-tagging.md)

1 LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations

Link: https://arxiv.org/pdf/2510.06657

Company: Google

Idea: speed-up (efficiency gains through quantization, model sharding, distillation, etc.)

Area: distillation + LLM

Overview:

This paper uses an LLM to tag short-video items and optimizes the pipeline around the LLM's inefficiency. Put simply, for a small set of high-quality videos, the LLM is used directly to generate tags for the short-video items, and knowledge distillation trains a student network that annotates the large mass of other videos.

Training the student network involves two steps:

(1) LLM annotation

For priority videos (new, popular, high-impact), the input includes sampled video frames, the description and a prompt. Multimodal models such as Gemini are used. Inference is optimized with model quantization, batch tuning and model sharding, raising throughput 2-3x.

The paper does not fine-tune the LLM; it gets the LLM to tag through prompting.

(2) Knowledge distillation

The LLM-generated annotations and probability scores are used as a "Silver Set" to train small "student" deep neural networks (DNNs). The student models use compact features such as precomputed video embeddings, are trained from scratch, and imitate the LLM's behavior. Cost and latency drop greatly with minimal quality loss.

My understanding is that the student network can be trained either periodically or with real-time incremental updates.

The resulting tags can be used later in retrieval and ranking.

Key insights:

(1) Inference optimizations such as model quantization, batch tuning and model sharding make LLM annotation feasible.

(2) Knowledge distillation is a powerful and necessary technique for extending validated, high-quality LLM annotations to the whole corpus. Using millions of LLM-derived examples to train smaller student DNNs on compact features achieves large-scale annotation throughput at a small fraction of the cost and latency of direct LLM inference, making the understanding of nuanced attributes broadly available.

Takeaways: (1) LLMs' understanding ability is very strong; without much work they can be deployed quickly and bring gains. (2) The distillation actually uses a DNN. Most people who use a DNN wouldn't dare say they used a DNN; they'd have to say they used a small LLM. (3) The paper is short and concise.

Caveats: using only their in-house LLM, Gemini, is understandable, but it would be better with more LLMs.

Credibility: production

Recommendation: practical value
