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
└── run_once.sh

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

## Mutable and immutable files

Mutable benchmark file:

- `autoresearch/experiments/restaurant_train.py`

The mutable file must define `build_policy()` and return a policy object with `decide_orders(...)` and optional `fit(...)`.

Immutable benchmark files:

- `autoresearch/experiments/restaurant_eval.py`
- `autoresearch/tasks.py`
- `program.md`
- `AGENTS.md`

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
