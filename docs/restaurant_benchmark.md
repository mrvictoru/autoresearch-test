# Restaurant Benchmark Contract

## Overview

The benchmark is a deterministic single-stock inventory simulation evaluated through `autoresearch/experiments/restaurant_eval.py`.

## Domain model

The immutable simulator lives in `autoresearch/tasks.py`.

- Demand per day is sampled from a Gaussian distribution and clamped to a non-negative integer.
- The policy chooses:
  - `reorder_point`
  - `target_stock`
- Daily cost includes:
  - replenishment cost
  - leftover waste cost
  - stockout penalty

The evaluator uses a fixed validation scenario:

- `days=30`
- `seed=42`

Primary objective:

- `score = -total_cost`
- higher score is better

Reported metrics:

- `score`
- `stockouts`
- `waste_units`
- `total_orders`

## Mutable and immutable boundaries

Mutable file:

- `autoresearch/experiments/restaurant_train.py`

Immutable files:

- `autoresearch/experiments/restaurant_eval.py`
- `autoresearch/tasks.py`
- `program.md`
- `AGENTS.md`

Do not change evaluator logic, benchmark mechanics, or output formatting during optimization.

## Evaluator contract

The evaluator must print:

- a `--- RESULTS ---` header
- one `key value` line per metric
- a `score` line
- a `METRIC_JSON:` line for automation

## Quick start

```bash
python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

```bash
./scripts/run_once.sh "adjust restaurant inventory policy"
```
