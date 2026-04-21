# Restaurant Benchmark Contract

## Overview

The restaurant benchmark in this repository is a deterministic inventory-control simulation implemented by `RestaurantInventoryTask` and evaluated through `autoresearch/experiments/restaurant_eval.py`.

Current scope is a **single-stock inventory policy** (not a multi-ingredient menu optimizer).

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

## What can and cannot change

### Can change

- How `get_model_state()` chooses `reorder_point` and `target_stock`
- Policy heuristics inside mutable train code
- Deterministic logic that maps context to those policy parameters

### Cannot change

- Evaluator parse/score contract
- Simulation internals in `RestaurantInventoryTask`
- Immutable benchmark files listed above

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
