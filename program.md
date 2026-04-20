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

For future restaurant harness experiments, mutable/immutable scope is defined by that experiment's contract (single mutable train file, immutable evaluator + benchmark data).

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

## Crash Recovery

1. Inspect recent log tail for root cause.
2. If failure is trivial, apply one fix and rerun once.
3. If still failing, mark run as crash and revert.

## Keep / Discard Rules

- Keep only strict improvements in primary `score`.
- On tie, use configured tie-break policy for the active experiment.
- Never keep a change that violates mutable/immutable boundaries.
