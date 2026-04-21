# Program

## Goal

Maximize the primary `score` reported by the immutable evaluator while preserving correctness and reproducibility.

## Setup

1. Read this file fully before any edits.
2. Run a baseline evaluation before making changes.
3. Keep all experiments reproducible (commit before run, record outputs).

## Experimentation Constraints

- Treat evaluator files and benchmark inputs as immutable.
- Only edit files explicitly listed as mutable for the active experiment.
- Keep changes small, testable, and reversible.
- Do not alter dependency declarations unless explicitly requested.

## Mutable / Immutable Files

For current neural mutation experiments:

- Mutable:
  - `autoresearch/experiments/neural_train.py`
- Immutable:
  - `autoresearch/experiments/neural_eval.py`
  - `autoresearch/mutation_runner.py`
  - `autoresearch/executor.py`
  - `research_brief.json`
  - `research_brief.yaml`

For restaurant harness experiments, use the fixed contract below.

For restaurant mutation experiments:

- Mutable:
  - `autoresearch/experiments/restaurant_train.py`
- Immutable:
  - `autoresearch/experiments/restaurant_eval.py`
  - `autoresearch/tasks.py`
  - `autoresearch/mutation_runner.py`
  - `autoresearch/executor.py`

## Restaurant Harness Contract

- Goal: maximize the restaurant inventory `score` on the fixed validation scenario exposed by `autoresearch/experiments/restaurant_eval.py`.
- Mutable code for this benchmark is limited to:
  - `autoresearch/experiments/restaurant_train.py`
- Treat the evaluator output on stdout as the source of truth for parsing `score`.
- Use `results.tsv` plus git commit history as the frontier ledger for keep/discard decisions.
- `scripts/run_once.sh` is the atomic harness helper for a single experiment attempt.
- Keep only strict score improvements over the current best `keep` row in `results.tsv`.

## Output & Logging Format

- Evaluators must print a stable parseable summary block to stdout:
  - header: `--- RESULTS ---`
  - one `key value` line per metric
  - include primary `score` line
- Evaluators may also print machine-readable metrics (for automation compatibility).

## Experiment Loop Protocol

1. Baseline run with unmodified mutable file.
2. Propose one focused mutation.
3. Commit before each run.
4. Run evaluator and parse `score`.
5. Keep change only if `score` improves; otherwise discard/revert.
6. Append run outcome to experiment history log.
7. Repeat until budget is exhausted.
8. Use `results.tsv`/`mutation_results.tsv` plus git branch/commit history as the optimization frontier ledger when running under a harness.
9. For restaurant harness runs, `scripts/run_once.sh` should perform the commit -> evaluate -> parse -> ledger update -> keep/discard flow.

## Crash Recovery

1. Inspect recent log tail for root cause.
2. Inspect the last 50 log lines before retrying.
3. If failure is trivial, apply one fix and rerun once.
4. If still failing, mark run as crash, append it to `results.tsv`, and revert.

## Keep / Discard Rules

- Keep only strict improvements in primary `score`.
- On tie, use configured tie-break policy for the active experiment.
- Never keep a change that violates mutable/immutable boundaries.
