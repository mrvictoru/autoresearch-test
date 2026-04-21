# autoresearch-test

`autoresearch-test` is a harness-only restaurant benchmark repository.

An external coding agent owns the optimization loop. This repository provides:

- immutable restaurant benchmark logic
- mutable experiment code for the active policy
- git-frontier helpers for keep/discard decisions
- a single-run harness helper script
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
- `autoresearch/tasks.py` — immutable restaurant simulation
- `autoresearch/experiments/restaurant_eval.py` — immutable evaluator entrypoint
- `autoresearch/experiments/restaurant_train.py` — mutable policy implementation
- `research_brief_restaurant.json` / `.yaml` — machine-readable benchmark contract

## Quick start

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run the immutable evaluator against the mutable policy:

```bash
python -m autoresearch.experiments.restaurant_eval \
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
