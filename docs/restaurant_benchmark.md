# Restaurant Benchmark Contract

## Overview

The restaurant benchmark in this repository is a deterministic inventory-control simulation implemented by `RestaurantInventoryTask` and evaluated through `autoresearch/experiments/restaurant_eval.py`.

Current scope is a **single-stock inventory policy**. The restaurant theme is represented as one aggregate inventory decision rather than a multi-ingredient, multi-recipe menu planner.

## Domain model

The evaluator simulates a fixed number of days with seeded stochastic demand.

- Demand per day is sampled from a Gaussian distribution (`demand_mean`, `demand_std`) and clamped to non-negative integers.
- A policy controls:
  - `reorder_point`
  - `target_stock` (internally clamped to at least `reorder_point + 1`)
- Daily outcomes accumulate:
  - stockouts (`stockout_penalty`)
  - leftover stock waste (`waste_cost`)
  - replenishment purchase cost (`unit_cost`)

The immutable restaurant evaluator currently uses a fixed validation setup of:

- `days=30`
- `seed=42`
- metrics emitted to stdout in the stable `--- RESULTS ---` block plus `METRIC_JSON`

Score is returned as:

- `score = -total_cost` (higher is better)

Additional metrics include `stockouts`, `waste_units`, and `total_orders`.

## Mutable vs immutable boundaries

For restaurant mutation experiments:

- Mutable:
  - `autoresearch/experiments/restaurant_train.py`
- Immutable:
  - `autoresearch/experiments/restaurant_eval.py`
  - `autoresearch/tasks.py`
  - `autoresearch/mutation_runner.py`
  - `autoresearch/executor.py`

Do not alter evaluator logic or benchmark cost/demand mechanics during optimization.

For harness-driven runs, only mutate `autoresearch/experiments/restaurant_train.py` and use `results.tsv` plus git history as the frontier ledger.

## What can and cannot change

### Can change

- How `get_model_state()` chooses `reorder_point` and `target_stock`
- Policy heuristics inside mutable train code
- Deterministic logic that maps context to those policy parameters
- Small reproducible code changes that can be committed, evaluated, and reverted by the harness

### Cannot change

- Evaluator parse/score contract
- Simulation internals in `RestaurantInventoryTask`
- Immutable benchmark files listed above
- The fixed validation seed/days exposed by `autoresearch/experiments/restaurant_eval.py`

## Quick-start evaluation

Run immutable evaluation with mutable restaurant train code:

```bash
python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

## Example mutation strategy patterns

- Reduce stockouts by raising `reorder_point` under high-demand assumptions.
- Reduce waste by lowering `target_stock` when stockouts remain acceptably low.
- Explore small paired adjustments (`reorder_point`, `target_stock`) and keep only strict score improvements.
- Use `scripts/run_once.sh` to atomically commit, evaluate, append to `results.tsv`, and automatically discard non-improving candidate commits.
