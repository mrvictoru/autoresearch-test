# Mutation Backend Guide

This guide covers the additive mutation backend for code-edit experiments.

## Why a second backend

- Keep parameter tuning (`AutoresearchRunner`) as default for simulator tasks.
- Use `MutationRunner` when experiments require code changes (optimizer family, architecture edits, training-loop changes).

## Immutable vs mutable boundary

- Immutable evaluator: `autoresearch/experiments/neural_eval.py`
- Mutable experiment code: `autoresearch/experiments/neural_train.py`
- Whitelist is enforced from `research_brief.yaml` (`allowed_mutable_files`).

## Research brief

`research_brief.yaml` (or `research_brief.json`) fields:

- `goal`
- `constraints`
- `allowed_mutable_files`
- `immutable_files`
- `time_budget_seconds`
- `tie_breaker_policy`

## Runner lifecycle

Each mutation iteration:

1. Propose structured patch/edits from `MutationAgent`
2. Apply in isolated `Workspace` candidate
3. Execute with `SafeExecutor` timeout/log capture
4. Parse score metrics
5. Classify as `keep`, `discard`, or `crash`
6. Promote candidate to frontier only on `keep`

## CLI usage

```bash
python -m autoresearch mutation \
  --task tiny_torch_classification \
  --brief research_brief.json \
  --agent-endpoint http://localhost:8080 \
  --agent-model local-model \
  --iterations 5
```

Artifacts are written under `artifacts/<task_name>/`:

- `mutation_results.csv`
- `mutation_experiments.json`
- `manifest.json`

## MutationRunner vs harness-driven loop

- **MutationRunner mode (Python-owned loop):**
  - The loop is orchestrated in-process by `MutationRunner`.
  - Iteration control, keep/discard/crash decisions, and artifact writing are handled inside the Python runtime.
  - Best for standard CLI runs (`python -m autoresearch mutation ...`).

- **Harness-driven mode (external-agent-owned loop):**
  - The outer coding harness owns planning, mutation application cadence, and run bookkeeping.
  - Python evaluators and benchmark logic remain the immutable scoring infrastructure.
  - Mutable scope is still constrained by the active experiment contract (`program.md`, `AGENTS.md`, and research brief boundaries).
