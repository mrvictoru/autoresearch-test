# autoresearch-test

`autoresearch-test` is a harness-only restaurant benchmark repository.

An external coding agent owns the optimization loop. This repository provides:

- immutable multi-item restaurant benchmark logic
- mutable experiment code for the active inventory policy
- neural network support via pre-installed `numpy` and `scikit-learn`
- git-frontier helpers for keep/discard decisions
- a single-run harness helper script
- post-run telemetry artifacts and a static HTML report generator
- documentation for the benchmark contract

Additional documentation:

- `docs/guide.md` for the harness workflow
- `docs/components.md` for file-level reference
- `docs/restaurant_benchmark.md` for the benchmark contract

## Repository layout

Core files:

- `program.md` — run protocol and mutable/immutable boundaries
- `AGENTS.md` — harness operating notes
- `scripts/run_once.sh` — atomic commit → evaluate → ledger → keep/discard helper
- `autoresearch/frontier.py` — git and `results.tsv` helpers
- `autoresearch/tasks.py` — immutable restaurant environment with menu overlap, perishability, lead times, and storage limits
- `autoresearch/experiments/restaurant_eval.py` — immutable evaluator entrypoint
- `autoresearch/experiments/restaurant_train.py` — mutable policy implementation via `build_policy()`
- `research_brief_restaurant.json` / `.yaml` — machine-readable benchmark contract

## Quick start

Run tests:

```bash
docker compose run --rm autoresearch python -m unittest discover -s tests -v
```

Run the immutable evaluator against the mutable policy:

```bash
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

Run one harness attempt:

```bash
./scripts/run_once.sh "adjust restaurant inventory policy"
```

That helper:

- commits tracked candidate changes
- runs the immutable restaurant evaluator
- parses `score` from stdout
- appends the attempt to `results.tsv`
- keeps only strict score improvements
- reverts discarded or crashed candidates

The benchmark score now reflects restaurant operating performance across fixed train and validation scenarios with overlapping ingredients, time-varying demand, perishability, supplier lead times, and storage constraints.

## Post-run report

Generate a trace artifact and interactive HTML report for a completed run:

```bash
docker compose run --rm autoresearch python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py \
  --report-dir artifacts/reports/latest
```

This writes:

- `artifacts/reports/latest/run_artifact.json` — per-order simulation trace plus aggregate metrics
- `artifacts/reports/latest/report.html` — static interactive replay and business report

Open `report.html` in a browser to inspect order flow, ingredient consumption, inventory movement, and business outcomes for the run.

If your Docker container cannot write back into the mounted workspace, point `--report-dir` at a writable container path such as `/tmp/report` for inspection inside the container, or run Docker with a user mapping that can write to the repo mount.

## Docker

Build the image:

```bash
docker compose build
```

Run the evaluator in Docker:

```bash
docker compose up autoresearch
```

Open a shell in Docker:

```bash
docker compose run --rm autoresearch bash
```

Run one harness attempt in Docker:

```bash
docker compose run --rm autoresearch ./scripts/run_once.sh "adjust restaurant inventory policy"
```
