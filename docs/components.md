# Autoresearch-Test — Component Reference

## Package overview

```text
autoresearch/
├── __init__.py
├── brief.py
├── control_plane.py
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

## `autoresearch/control_plane.py`

Local multi-agent orchestration helpers layered around the immutable benchmark.

Main APIs:

- `init_control_plane(...)`
- `add_experiment_idea(...)`
- `review_experiment_ideas(...)`
- `launch_experiment_worker(...)`
- `run_experiment_worker(...)`
- `promote_experiment_worker(...)`
- `cleanup_experiment_worker(...)`
- `summarize_control_plane(...)`

Responsibilities:

- maintain repo-local campaign state in `research/state/`
- deduplicate planner hypotheses against prior ideas and memory notes
- create isolated git worktrees for experiment workers
- run one worker attempt while keeping `results.tsv` as the shared frontier
- report active/completed worker state and the current best score

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

## `scripts/control_plane.sh`

Thin shell wrapper for `python -m autoresearch.control_plane`.

Responsibilities:

- initialize campaign state
- add/review ideas
- launch workers
- summarize status
- promote or clean up worker branches

## `scripts/run_worker.sh`

Worker-facing wrapper for the `run-worker` control-plane command.

Responsibilities:

- validate that only the mutable benchmark file changed
- commit the candidate in the worker worktree
- run the Docker evaluator
- update `results.tsv`
- revert discarded/crashed candidates
