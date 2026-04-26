# Program

## Goal

Maximize the primary `score` reported by the immutable restaurant evaluator while preserving correctness, reproducibility, and benchmark integrity.

The benchmark score is the only value used for keep/discard decisions. Auxiliary metrics are for diagnosis only.

## Setup

1. Read this file fully before any edits.
2. Run a baseline restaurant evaluation before making changes.
3. Keep all experiments reproducible.

4. Establish a research branch for the session before iterating.
5. Initialize `results.tsv` if it does not already exist.

## Agent Initialization

1. Choose a short run tag for the session.
2. Create or confirm a branch named `autoresearch/<tag>`.
3. Read the in-scope files: `program.md`, `AGENTS.md`, `README.md`, `docs/guide.md`, and the benchmark files.
4. Run the baseline evaluator before modifying the mutable policy.
5. Record the baseline in `results.tsv`.
6. Confirm the setup is valid before starting autonomous iterations.

## Repository mode

This repository is harness-only.

The active benchmark is an ambitious restaurant inventory environment with menu items sharing ingredients, time-varying demand, ingredient perishability, supplier lead times, and storage constraints.

Once the autonomous loop begins, do not pause to ask for permission to continue unless the user explicitly interrupts.

## Mutable / Immutable Files

Mutable:

- `autoresearch/experiments/restaurant_train.py`

Immutable:

- `autoresearch/experiments/restaurant_eval.py`
- `autoresearch/tasks.py`
- `program.md`
- `AGENTS.md`

All other files are read-only unless explicitly authorized.

## Harness Contract

- Treat evaluator stdout as the source of truth for parsing `score`.
- Use `results.tsv` plus git commit history as the frontier ledger.
- `scripts/run_once.sh` is the atomic helper for one experiment attempt.
- Keep only strict score improvements over the current best kept run.
- The mutable policy file must expose `build_policy()`.
- The immutable evaluator and benchmark files must not be changed during normal research iterations.
- Every experiment attempt must be logged, including keep, discard, and crash outcomes.

## Output format

Evaluator output must include:

- header: `--- RESULTS ---`
- one `key value` line per metric
- a primary `score` line
- service and cost metrics for diagnosis
- `METRIC_JSON` for automation compatibility

The evaluator should be able to run unattended from Docker and should write report artifacts only when explicitly requested.

## Experiment loop

1. Inspect the latest commit and the current best score in `results.tsv`.
2. Edit only `autoresearch/experiments/restaurant_train.py`.
3. Commit the candidate state.
4. Run the immutable evaluator.
5. Parse `score` from stdout.
6. Append the attempt to `results.tsv`.
7. Keep only strict improvements; otherwise revert.
8. Repeat the loop without asking the human whether to continue.

Within `autoresearch/experiments/restaurant_train.py`, you may change the policy logic, features, heuristics, and optional training procedure. Do not change evaluator or benchmark files during normal research iterations.

## Experiment Execution

Preferred execution path:

```bash
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
	--experiment autoresearch/experiments/restaurant_train.py
```

For a single atomic attempt that handles commit, evaluate, log, and revert behavior, prefer:

```bash
./scripts/run_once.sh "short description of the candidate change"
```

Read the evaluator output from the log or terminal capture after the run completes. Do not rely on interactive output during the experiment.

## Decision Making

If the new `score` is strictly better than the current best kept score, keep the candidate.

If the new `score` is not better, revert the candidate commit and mark the attempt as `discard`.

If the run fails twice with the same candidate state, mark the attempt as `crash`, record the failure in `results.tsv`, and revert.

Crash recovery should be conservative: fix obvious typos or import mistakes and retry once; otherwise move on.

## Continuous Operation

Once the experiment loop begins, keep iterating until manually stopped.

The expected operating model is unattended autonomous research, not a single-shot manual edit.

## Exploration / Creativity on Plateau

### Detecting a plateau

Treat the current run as plateaued when **any** of the following is true:

- No strict score improvement has been recorded for **5 or more consecutive kept attempts**.
- The same candidate change has been tried (or an equivalent one) in **3 or more discarded attempts** without a new hypothesis.
- Every recent attempt ends in a `crash` or `discard` with no clear direction for recovery.
- Auxiliary diagnostics (service level, waste, stockout penalty) have all stopped moving in the desired direction across the last 5 attempts.

### Permission to explore divergent strategies

When a plateau is detected, **you are explicitly permitted — and encouraged — to try substantially different approaches**, even if they depart from the current best policy design.
Do not keep refining the same parameter tweaks indefinitely.
Treat the plateau as a signal to restructure the hypothesis space.

### Concrete experimental tactics to try

Try these tactics in roughly this order of increasing invasiveness; move to the next when the current class of changes stops yielding improvement:

1. **Hyperparameter search** — sweep `safety_factor`, `freshness_bias`, `recent_demand_weight`, `hidden_layer_sizes`, `safety_margin`, oracle look-ahead window, learning rate, and regularization strength over a structured grid or random sample.
2. **Hybrid rule-based + neural policy** — combine the `AdaptiveRestaurantPolicy` heuristic as a fallback or blended signal inside a neural wrapper: e.g. use the rule-based order as a prior and let a learned residual correct it. This captures the best of both worlds.
3. **Alternative model architectures** — replace the MLP with gradient-boosted trees (sklearn `GradientBoostingRegressor`), a linear model with polynomial features, or a simple recurrent structure over the last N days of usage history.
4. **Feature engineering** — add interaction terms, rolling statistics (mean, std over a sliding window), day-of-week × ingredient cross features, demand-trend slope, and capacity-pressure signals.
5. **Alternative training targets** — instead of oracle future demand, train on actual fulfilled quantities, waste-weighted shortfall, or a composite reward that penalises both stockout and waste.
6. **Reward / cost shaping** — re-weight the order cost, holding cost, waste cost, and stockout penalty multipliers inside the policy's loss signal to steer learning toward the benchmark's scoring formula.
7. **Ablation studies** — isolate the contribution of individual components (e.g. disable freshness cap, remove weekday splitting, remove pipeline term) to understand which features drive score vs. which add noise.
8. **Ensemble / voting policies** — run two or more policies in parallel (e.g. rule-based and neural) and blend their order suggestions, selecting the higher-confidence recommendation per ingredient.
9. **Different search strategies** — instead of greedy keep-best, try a brief simulated-annealing step: accept a slightly worse score with low probability to escape a local optimum.

### Guardrails during creative exploration

- **Log every attempt** in `results.tsv`, including exploratory ones. Use the `explore` outcome tag (instead of `discard`) when an attempt is intentionally experimental rather than a straightforward incremental change, so plateau-breaking runs can be distinguished from normal iterations. An exploratory attempt that strictly improves the score should still be recorded as `keep`.
- **Keep changes reversible** — each candidate must be a single commit that can be reverted cleanly. Do not stack unreviewed changes across multiple commits before evaluating.
- **Timebox each tactic** — spend no more than **3 attempts** on any single new tactic before either adopting it (if it improved score) or moving on to the next tactic in the list above.
- **Do not touch immutable files** — the evaluator, benchmark, and task definitions remain off-limits even during creative exploration. All changes stay within `autoresearch/experiments/restaurant_train.py`.
- **Preserve the harness contract** — `build_policy()` must always return a valid `RestaurantPolicy` object. Every exploratory policy must pass the evaluator without crashing before being kept.
- **Return to stable baseline if exploration fails** — if none of the tactics yield improvement after exhausting the list, revert to the best-known kept commit and document the plateau in `results.tsv` with an `explore_exhausted` note.

## Crash recovery

1. Inspect the recent log tail.
2. Retry once if the failure is trivial.
3. If it still fails, record a `crash` row in `results.tsv` and revert.

Common benchmark failures include invalid order outputs, capacity violations, and logic errors in the mutable policy.

Typical crash causes include import errors, invalid policy outputs, impossible capacity allocations, and exceptions inside the mutable policy or evaluator.
