# Copilot Instructions

## Build and test commands

Use the Docker service as the canonical execution environment:

```bash
docker compose build
docker compose run --rm autoresearch python -m unittest discover -s tests -v
docker compose run --rm autoresearch python -m unittest tests.test_autoresearch.AutoresearchFrameworkTests.test_restaurant_eval_smoke_test_current_mutable_policy -v
docker compose run --rm autoresearch python -m unittest tests.test_control_plane.ControlPlaneTests.test_run_worker_records_frontier_and_reverts_discarded_candidate -v
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py \
  --report-dir artifacts/reports/latest
docker compose run --rm autoresearch ./scripts/run_once.sh "short experiment note"
```

If `docker compose run` is blocked by an NVIDIA runtime mismatch in the host environment, a plain `docker run` fallback may be used for evaluation, but the normal workflow continues to prefer `docker compose run`.

## High-level architecture

This repository is a harness-only benchmark. The benchmark itself is immutable and lives in `autoresearch/tasks.py`; it defines deterministic training and validation scenarios for a restaurant inventory simulation with overlapping ingredient usage, perishability, supplier lead times, and both per-ingredient and global storage constraints.

`autoresearch/experiments/restaurant_eval.py` is the immutable evaluator entrypoint. It dynamically imports the mutable experiment file, calls `build_policy()`, calls `fit(training_scenarios, task)` if the returned policy provides it, evaluates on validation scenarios, prints a stable `--- RESULTS ---` block plus `METRIC_JSON`, and optionally writes `run_artifact.json` and `report.html` through `autoresearch/reporting.py`.

`autoresearch/experiments/restaurant_train.py` is the active policy surface. The current file uses a small registry so `build_policy()` can switch between named policies, with the default path returning the neural-network policy and the adaptive heuristic as the fallback. Future policy work should stay behind that `build_policy()` contract instead of changing the evaluator.

The harness/frontier layer is split across `autoresearch/frontier.py`, `scripts/run_once.sh`, `autoresearch/control_plane.py`, `scripts/control_plane.sh`, and `scripts/run_worker.sh`. `frontier.py` owns branch creation, commit/revert helpers, and `results.tsv`; `run_once.sh` packages one attempt into commit -> evaluate -> log -> keep/discard; the control plane adds idea review, worker manifests in `research/state/`, isolated worktrees under `artifacts/control-plane/worktrees/`, and shared frontier updates.

## Key conventions

- Treat this as a benchmark harness, not a general app: for benchmark work, only `autoresearch/experiments/restaurant_train.py` is mutable unless the user explicitly broadens scope.
- Do not change `autoresearch/experiments/restaurant_eval.py`, `autoresearch/tasks.py`, `program.md`, or `AGENTS.md` during normal optimization work.
- `build_policy()` must return an object with `decide_orders(...)` and optional `fit(...)`.
- `results.tsv` is the optimization ledger. Every attempt is logged, but only `decision=keep` rows count when reading the current best score.
- Preserve evaluator output shape. Automation depends on the `--- RESULTS ---` block and the `METRIC_JSON:` line.
- The normal workflow is container-first: repo docs and the control plane assume Python entrypoints run via `docker compose run --rm autoresearch ...`.
- `scripts/run_once.sh` and worker runs only keep strict score improvements; discarded or crashed candidates are reverted after logging.
- Worker evaluation is intentionally narrow: `run_worker` validates that only the mutable benchmark file changed in the worker worktree.
- Report generation is post-run analysis only. `--report-dir` writes a replay artifact bundle without changing the scoring path.
- The default mutable policy relies on `numpy` and `scikit-learn`, which the docs expect to be available in the Docker image.
