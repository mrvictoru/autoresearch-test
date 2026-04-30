# Restaurant Benchmark Contract

## Overview

The benchmark is an immutable multi-item restaurant inventory environment evaluated through `autoresearch/experiments/restaurant_eval.py`.

## Domain model

The immutable simulator lives in `autoresearch/tasks.py`.

The environment includes:

- menu items with overlapping ingredient usage
- lunch and dinner demand periods
- weekday and late-horizon demand shifts
- ingredient perishability through shelf-life buckets
- supplier lead times through an incoming-order pipeline
- per-ingredient and total storage constraints

The mutable policy chooses per-ingredient order quantities each day.

The mutable experiment file must expose `build_policy()` returning an object with:

- `decide_orders(observation)`
- optional `fit(training_scenarios, task)`

The evaluator uses deterministic train and validation scenario sets generated from the benchmark seed.

Primary objective:

- `score = revenue - order_cost - holding_cost - waste_cost - stockout_penalty`
- higher score is better

Reported metrics:

- `score`
- `service_level`
- `revenue`
- `fulfilled_orders`
- `lost_orders`
- `waste_units`
- `waste_cost`
- `holding_cost`
- `order_cost`
- `stockout_penalty`

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
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

```bash
./scripts/run_once.sh "adjust restaurant inventory policy"
```

Optional multi-agent control-plane flow:

```bash
./scripts/control_plane.sh init --max-workers 2
./scripts/control_plane.sh review
./scripts/control_plane.sh launch-worker --idea-id <idea-id>
./scripts/run_worker.sh --worker-id <worker-id> --message "one hypothesis"
```

This wraps the same immutable evaluator with isolated git worktrees, shared `results.tsv` frontier tracking, and checked-in role contracts under `research/agents/`.
