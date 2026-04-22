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

## Crash recovery

1. Inspect the recent log tail.
2. Retry once if the failure is trivial.
3. If it still fails, record a `crash` row in `results.tsv` and revert.

Common benchmark failures include invalid order outputs, capacity violations, and logic errors in the mutable policy.

Typical crash causes include import errors, invalid policy outputs, impossible capacity allocations, and exceptions inside the mutable policy or evaluator.
