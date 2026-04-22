# Autoresearch-Test — Component Reference

## Package overview

```text
autoresearch/
├── __init__.py
├── brief.py
├── frontier.py
├── reporting.py
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
- `write_report_bundle(...)`
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

## `autoresearch/reporting.py`

Static post-run visualization and report generation.

Main API:

- `write_report_bundle(output_dir, artifact)`

Responsibilities:

- write `run_artifact.json`
- render `report.html`
- provide a CLI for rebuilding reports from saved artifacts

## `autoresearch/tasks.py`

Contains the immutable restaurant simulation.

Main type:

- `RestaurantInventoryTask`

The task produces:

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

It can also emit deterministic per-order telemetry and checkpoints for post-run analysis.

Telemetry includes customer-order events, restock orders and arrivals, spoilage, holding cost, inventory snapshots, and cumulative cash for post-run replay.

## `autoresearch/experiments/restaurant_eval.py`

Immutable evaluator entrypoint.

Responsibilities:

- load the mutable train module
- call `build_policy()`
- optionally fit the returned policy
- score it with `RestaurantInventoryTask`
- print the stable `--- RESULTS ---` block
- print `METRIC_JSON` for automation compatibility
- optionally write `run_artifact.json` and `report.html`

The evaluator can run in score-only mode or in report mode via `--report-dir`.

## `autoresearch/experiments/restaurant_train.py`

Mutable benchmark file.

Responsibilities:

- define `build_policy()`
- return the policy under evaluation

The current contract expects a policy object with `decide_orders(...)` and optional `fit(...)`.

## `scripts/run_once.sh`

Atomic helper for one harness attempt.

Responsibilities:

- initialize `results.tsv` if needed
- commit candidate changes
- run the evaluator
- record `keep`, `discard`, or `crash`
- revert discarded/crashed commits
