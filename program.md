# Program

## Goal

Maximize the primary `score` reported by the immutable restaurant evaluator while preserving correctness, reproducibility, and benchmark integrity.

## Setup

1. Read this file fully before any edits.
2. Run a baseline restaurant evaluation before making changes.
3. Keep all experiments reproducible.

## Repository mode

This repository is harness-only.

The active benchmark is an ambitious restaurant inventory environment with menu items sharing ingredients, time-varying demand, ingredient perishability, supplier lead times, and storage constraints.

## Mutable / Immutable Files

Mutable:

- `autoresearch/experiments/restaurant_train.py`

Immutable:

- `autoresearch/experiments/restaurant_eval.py`
- `autoresearch/tasks.py`
- `program.md`
- `AGENTS.md`

## Harness Contract

- Treat evaluator stdout as the source of truth for parsing `score`.
- Use `results.tsv` plus git commit history as the frontier ledger.
- `scripts/run_once.sh` is the atomic helper for one experiment attempt.
- Keep only strict score improvements over the current best kept run.
- The mutable policy file must expose `build_policy()`.

## Output format

Evaluator output must include:

- header: `--- RESULTS ---`
- one `key value` line per metric
- a primary `score` line
- service and cost metrics for diagnosis
- `METRIC_JSON` for automation compatibility

## Experiment loop

1. Edit only `autoresearch/experiments/restaurant_train.py`.
2. Commit the candidate state.
3. Run the immutable evaluator.
4. Parse `score` from stdout.
5. Append the attempt to `results.tsv`.
6. Keep only strict improvements; otherwise revert.

Within `autoresearch/experiments/restaurant_train.py`, you may change the policy logic, features, heuristics, and optional training procedure. Do not change evaluator or benchmark files during normal research iterations.

## Crash recovery

1. Inspect the recent log tail.
2. Retry once if the failure is trivial.
3. If it still fails, record a `crash` row in `results.tsv` and revert.

Common benchmark failures include invalid order outputs, capacity violations, and logic errors in the mutable policy.
