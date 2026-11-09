# Recommendation Algorithms (2025.8.13): Spotify's helper for model development

> [中文版](2025-08-13-spotify-llm-judge.md)

1 Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge

Link: https://arxiv.org/pdf/2508.08777

Company: Spotify

Area: LLM as a judge

Overview:

This paper uses an LLM to evaluate whether a model under development suits users' tastes. That is, the LLM evaluates and tests the model, with a focus on evaluating user satisfaction, so the model can be iterated and optimized quickly.

It does not directly evaluate the recommendation quality of a (user, item) pair. Instead it generates a user profile from the user behavior sequence and then evaluates. This shows the importance of the user behavior sequence, which matters much more than the user's meta features. So there are two big steps: profile extraction and evaluation.

(1) Profile extraction

* Input: the user's listening history over the last 90 days (e.g. titles, descriptions, transcripts and topic tags of episodes listened to, and completion rates).

* Output: a structured natural-language profile (in JSON) capturing two dimensions:

1) Content preferences: topical focus (e.g. space, technology), entities (e.g. specific hosts), exploration vs. specialization (broad vs. narrow topics), cross-domain curiosity.

2) Listening patterns: habits (e.g. high completion rate), depth (sampling vs. deep consumption), format preference (structured discussion vs. casual chat).

(2) Evaluation

Input: the user profile mined in (1) + metadata of the candidate episode (title, description, tags).

LLM prompting: zero-shot, with Chain-of-Thought reasoning about the match.

To verify the evaluation quality, some users were recruited as study participants to check the evaluations, in order to improve the evaluator.

Human validation of the evaluator: overall it works well and agrees closely with human judgment.

Takeaways:

(1) Using an LLM directly to evaluate how well a model under development works, without user surveys, is a good idea. Especially in an era that cares not only about consumption metrics but even more about user satisfaction.

(2) It is hard to see user satisfaction in an online A/B test. This is a good complement to it, and also a good complement to user surveys.

(3) Understanding is the LLM's strength, and this result puts that into practice.

Credibility: production

Recommendation: practical value
