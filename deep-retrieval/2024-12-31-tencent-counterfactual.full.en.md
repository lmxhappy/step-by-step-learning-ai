# Recommendation algorithm papers

> [中文版](2024-12-31-tencent-counterfactual.md)

1 Unsupervised dense retrieval with conterfactual contrastive learning

Link: https://arxiv.org/pdf/2412.20756

Company: Tencent

Area: retrieval + counterfactual

Overview:

The paper proposes an unsupervised dense retrieval method that uses counterfactual contrastive learning to make the model more sensitive to fine-grained relevance signals. It uses Shapley values from game theory to extract key passages, and trains the model on counterfactual samples so it can tell modifications to key passages apart from modifications to non-key passages. Experiments show the method can extract key passages without relying on passage-level annotation, improves the model's robustness to adversarial attacks, and beats existing methods. The work offers a new direction for building more interpretable and robust dense retrieval models.

Counterfactual samples are samples obtained by modifying positive samples. They include partial counterfactual samples, full counterfactual samples, adversarial counterfactual samples, and so on. Adversarial counterfactual samples are built by randomly replacing a certain fraction of the words in the original document, then picking the modified version that changes the model's relevance score the most, to simulate a malicious attack.

Credibility: offline

Recommendation: worth a look if you have time
