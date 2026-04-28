# Autoresearch-Test — Harness Guide

## Overview

This repository supports a single workflow: an external harness iteratively edits `autoresearch/experiments/restaurant_train.py` and evaluates it against an immutable restaurant benchmark.

The benchmark models:

- menu items sharing ingredients
- lunch and dinner demand periods
- weekday and late-horizon demand shifts
- ingredient perishability
- supplier lead times
- per-ingredient and total storage limits

## Architecture

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

scripts/
├── control_plane.sh
├── run_once.sh
└── run_worker.sh

program.md
AGENTS.md
research_brief_restaurant.json
research_brief_restaurant.yaml
```

## Harness workflow

1. Read `program.md`.
2. Run a baseline evaluation.
3. Edit only `autoresearch/experiments/restaurant_train.py`.
4. Commit the candidate state.
5. Run the immutable evaluator.
6. Parse `score` from stdout.
7. Append the attempt to `results.tsv`.
8. Keep only strict improvements; otherwise revert.

`./scripts/run_once.sh` packages steps 4–8 into one atomic helper.

For multi-agent operation, the repo also provides a local control plane:

1. `./scripts/control_plane.sh init --max-workers N` creates repo-local state.
2. Planner ideas are recorded with `add-idea` and screened with `review`.
3. `launch-worker` creates one isolated git worktree per approved idea.
4. The experiment worker edits only `autoresearch/experiments/restaurant_train.py` inside that worktree.
5. `./scripts/run_worker.sh --worker-id ... --message ...` commits, evaluates, records the result in the shared `results.tsv`, and reverts discarded or crashed candidates.
6. `status` summarizes active workers, recent results, and the best kept frontier entry.

## Copilot CLI workflow

You can use Copilot CLI as the outer harness that drives the repository loop. The CLI does not replace the benchmark; it coordinates edits, evaluation, and frontier management while the repo itself stays responsible for the immutable evaluator and policy contract.

End-to-end flow:

1. Open the repository in your shell and start your Copilot CLI session.
2. Read `program.md` and this guide first.
3. Inspect `autoresearch/experiments/restaurant_train.py`, then edit only that file.
4. Run the benchmark in Docker:

```bash
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

5. If the score improves, keep the change; otherwise revert it.
6. Use `./scripts/run_once.sh "your short experiment note"` when you want the repo to handle commit, run, parse, and keep/discard automatically.
7. Repeat the loop, changing only the mutable policy file until the score stops improving.

Typical Copilot CLI responsibilities in this workflow:

- read the repo instructions and benchmark contract
- propose or apply edits to `autoresearch/experiments/restaurant_train.py`
- run the Docker evaluator command above
- compare the new score against `results.tsv`
- decide whether to keep the candidate commit or revert it

Typical repo responsibilities in this workflow:

- define the benchmark in `autoresearch/tasks.py`
- enforce the immutable evaluator contract in `autoresearch/experiments/restaurant_eval.py`
- record frontier history in `results.tsv`
- provide the atomic helper in `scripts/run_once.sh`
- provide local control-plane coordination in `autoresearch/control_plane.py`

## Mutable and immutable files

Mutable benchmark file:

- `autoresearch/experiments/restaurant_train.py`

The mutable file must define `build_policy()` and return a policy object with `decide_orders(...)` and optional `fit(...)`.

Immutable benchmark files:

- `autoresearch/experiments/restaurant_eval.py`
- `autoresearch/tasks.py`
- `program.md`
- `AGENTS.md`

## Neural network support

The Docker image ships with `numpy` and `scikit-learn` pre-installed.
`restaurant_train.py` therefore supports MLP-based policies out of the box.

`NeuralNetworkPolicy` (the default) trains a `sklearn.neural_network.MLPRegressor`
on oracle-labelled rollouts generated from training-scenario frames:

1. **Feature extraction** — cyclic day-of-week encoding, storage utilisation,
   per-ingredient inventory / pipeline ratios, lead time, shelf life, cost signals.
2. **Oracle labels** — for each training day the look-ahead demand over the
   next `lead_time + 2` days is computed from future scenario frames; the
   resulting order quantities become training targets.
3. **Capacity post-processing** — MLP raw predictions are clipped to respect
   per-ingredient and total storage hard limits before being returned.

To switch policies, change the argument in `build_policy()`:

```python
# MLP policy (default)
return REGISTRY.build("neural_network")

# Rule-based heuristic fallback
return REGISTRY.build("adaptive")

# Custom MLP architecture
return REGISTRY.build("neural_network", hidden_layer_sizes=(256, 128, 64))
```

To add an entirely new policy class, register it with the `REGISTRY`:

```python
REGISTRY.register("my_policy", MyCustomPolicy)
return REGISTRY.build("my_policy")
```

All registered policies must implement `decide_orders(observation)` and may
optionally implement `fit(scenarios, task)` for supervised pre-training.

## Machine-readable brief

`research_brief_restaurant.json` and `research_brief_restaurant.yaml` provide the same restaurant contract in machine-readable form:

- goal
- evaluator command
- allowed mutable files
- immutable files
- time budget
- tie-breaker policy

## Validation

Run the evaluator directly:

```bash
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

Run the test suite:

```bash
docker compose run --rm autoresearch python -m unittest discover -s tests -v
```

Generate a post-run artifact bundle and static HTML report:

```bash
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py \
  --report-dir artifacts/reports/latest
```

Initialize and inspect the control plane:

```bash
./scripts/control_plane.sh init --max-workers 2
./scripts/control_plane.sh status
```

This writes a deterministic `run_artifact.json` trace and a browser-openable `report.html` bundle for inspecting order flow, ingredient consumption, inventory behavior, and business metrics after the simulation completes.

If Docker cannot write into the mounted repository on your machine, use a writable path such as `/tmp/report` for a one-off inspection inside the container, or run Docker with a user mapping that can write to the workspace mount.

## Visualizer workflow

The saved run artifact is the source of truth for the visualizer. After a completed run, the report reader uses `run_artifact.json` to animate the event log and reconstruct the business state over time.

What the visualizer shows:

- individual customer orders as they arrive and resolve
- which ingredients each order consumed when it was fulfilled
- which ingredients caused a lost sale when stock was insufficient
- inventory bars and aging/spoilage behavior across the run
- cash flow, order cost, holding cost, waste cost, and stockout penalty trends
- menu economics and ingredient-level business summaries

How to use it:

1. Run the evaluator with `--report-dir` so the artifact bundle is written.
2. Open `report.html` in a browser.
3. Use play, pause, and step controls to animate the log.
4. Scrub the timeline to jump between events and checkpoints.
5. Switch ingredients to inspect inventory history for a specific item.

The visualizer is intentionally post-run analysis only. It does not affect the benchmark score or the keep/discard frontier.

## Docker

The Docker image is configured for the same harness-only workflow.

Default container behavior:

```bash
docker compose up autoresearch
```

This runs the immutable restaurant evaluator against the mutable train file.

For interactive work:

```bash
docker compose run --rm autoresearch bash
```
