Meta hands the ads-ranking tuning loop to an agent. Humans gate five checkpoints. Iterations per engineer go up several times over.

Paper: Agentic ML Exploration (A-MLE) for Ads Ranking, Meta (https://arxiv.org/abs/2609.08248)

The bottleneck in ads ranking isn't capacity or compute - it's how many loops one human can push. Idea, code, train, read metrics, conclude: a senior engineer finishes single digits a week, and Meta's hundreds of long-tail ranking models never reach an expert's calendar.

A-MLE's bet: a general LLM can already run that loop, and what's missing is scaffolding - a skill library that really reads configs, edits code, submits jobs and runs evals, plus a knowledge base of what's been tried where.

1 - Five stages

Hypothesis. Reads recent configs, the baseline, and the rolling history of what was tried on this model; analyzers and a literature retriever feed in, a second LLM scores candidates on novelty and feasibility. Hypotheses must track the model's live state: ranking models change daily, last week's diagnosis is stale.

Strategy. Explore/exploit sequencing under a fixed budget. "Burn compute or play safe" is a value judgment, so a human calls it.

Execution. Sandboxed edits, tests, job submission, log polling. The critical skill: telling infra failure from training divergence - dead machine means retry, exploded loss means rethink, and misjudging costs days of compute.

Analysis. Significance against a rolling baseline, not a frozen snapshot - production keeps refreshing, so an old snapshot turns drift into a fake win.

Knowledge base. The only cross-model state: a versioned markdown tree in the repo recording which technique was tried on which model and how it went. Kickoff pulls what won on similar models; results, failures included, are written back.

2 - Results

No online A/B - offline only, best config +2.56%, QPS-neutral. The real claim is throughput, several times the human baseline. Of three tiers - manual, semi-auto (humans propose, scripts execute), full A-MLE - the middle gains least. Leave one handoff in the loop and human scheduling is the bottleneck again: the leverage is in wiring the whole chain, not in point tools. And the biggest gains come from transfer, not new techniques: porting what won on model A onto B, C, D. Humans can do this, they just can't finish the list.

Takeaways

- Alibaba's Astar bets the opposite way: train a dedicated model to propose directions. Not a conflict, a sequencing question - an agent with the domain skill library scores 68% on basic capability tests, a generic ML agent 16%. Scaffolding comes first.
- Transfer is the win, and it needs a structured record of technique-by-model outcomes. Most teams have none; it's scattered across experiment platforms, reports and chat logs. That record pays off with humans querying it too, and you can start building it today.

Credibility: real Meta ranking models, offline only, zero A/B.

#RecSys #MachineLearning #LLM #AIAgents #AI
