# Autoresearch-Test — Component Reference

## Package overview

```text
autoresearch/
├── __init__.py
├── brief.py
├── frontier.py
├── tasks.py
└── experiments/
   ├── __init__.py
   ├── restaurant_eval.py
   └── restaurant_train.py
```

## `autoresearch/__init__.py`

Defines the public harness-facing package surface.

Exports:

- `ResearchBrief`
- `RestaurantInventoryTask`
- frontier helpers for branch, commit, revert, and `results.tsv`
- `load_research_brief(...)`

## `autoresearch/brief.py`

Loads machine-readable benchmark contracts from JSON or YAML.

Main API:

- `ResearchBrief`
- `load_research_brief(path)`

## `autoresearch/frontier.py`

Git and ledger helpers used by the harness.

Main APIs:

- `create_research_branch(tag)`
- `commit_before_run(message)`
- `revert_last_commit()`
- `get_current_sha()`
- `init_results_tsv(path)`
- `append_result(...)`
- `read_best_result(path)`

## `autoresearch/tasks.py`

Contains the immutable restaurant simulation.

Main type:

- `RestaurantInventoryTask`

The task produces:

- `score`
- `stockouts`
- `waste_units`
- `total_orders`

## `autoresearch/experiments/restaurant_eval.py`

Immutable evaluator entrypoint.

Responsibilities:

- load the mutable train module
- call `get_model_state()`
- score it with `RestaurantInventoryTask`
- print the stable `--- RESULTS ---` block
- print `METRIC_JSON` for automation compatibility

## `autoresearch/experiments/restaurant_train.py`

Mutable benchmark file.

Responsibilities:

- define `get_model_state()`
- return the policy under evaluation

## `scripts/run_once.sh`

Atomic helper for one harness attempt.

Responsibilities:

- initialize `results.tsv` if needed
- commit candidate changes
- run the evaluator
- record `keep`, `discard`, or `crash`
- revert discarded/crashed commits
