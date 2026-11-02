At YouTube, adding a new ranking task went from a code change to a config change. Experiment cycle: 24 days to 11.

Paper: Lightweight Ranking Heads, Google (https://arxiv.org/abs/2609.25433)

Adding a prediction target to a ranker sounds like adding a head. It stalls at two scales. On the model: the new head's gradients reach the shared layers, so the backbone cold-starts and existing tasks regress from negative transfer. On the fleet: dozens of downstream models consuming that score no longer line up, each needing its own retrain - months of fragmentation where nobody dares launch. The real cost was never training, it was the queue.

1 - The shallow tower

A few hidden layers at the same level as the main heads, on the same shared representation. Stop-gradients at the bottom are the crux: backprop updates only the light head's own layers, the shared ones never move. The backbone is oblivious - no cold start, no negative transfer. Ablation confirms this is a precondition, not a nicety: drop the stop-gradients and main-head P(CTR) AUC falls 0.7767 to 0.7704.

It is also not checkpointed, but reset every run. Counterintuitive, but it removes state management: no versions, no compatibility, detach means detached. The cost is a metric dip that recovers in a few steps.

2 - Central configuration

That fixes one model. The fleet needs task definitions pulled out of model code - labels, losses, activations, metrics externalized - so one new task injects into every experimenting model at once, no per-model edits.

3 - Production safeguards

Dynamic injection will fail in production: config hasn't propagated, slots are stale, some model never picked the head up. Three guards: serving-time defaults for a missing head, quality gates before export, a dashboard surfacing outlier heads. Plus one easily missed design - config is read once at the start of a run and frozen, keeping stale models safe and results interpretable.

Results

Joint experiment start: 17 days to 5. Full cycle: 24 to 11.

One win worth noting: a light head serving only paid subscribers of YouTube's third-party channels lifted their engagement +13.83% - a small-slice experiment that would never have cleared the old queue.

Takeaways

- The contribution is treating experiment organization as the optimizable object, not shipping another model tweak. And the half that matters is central config, not the shallow tower: light heads alone save only single-model time.
- Stop-gradient plus no persistence is a read-only probe on the backbone. Nothing depends on YouTube's scale - any shared-bottom multi-task ranker can copy it.
- This is a backend config center applied to model training: adding a task goes from "edit code and deploy" to "edit config and push," with matching fallbacks, gates and monitoring. The inversion: microservice config wants hot updates, while here a mid-run change would make results uninterpretable, so it freezes.

#RecSys #RecommenderSystems #MachineLearning #MLOps #AI
