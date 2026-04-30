# Experiment worker

Operate inside one isolated git worktree.

Rules:
- edit only `autoresearch/experiments/restaurant_train.py`
- implement exactly one hypothesis per run
- after the evaluator completes, inspect policy behavior before deciding keep/discard:
  - identify which ingredients caused the most stockouts and lost orders
  - identify which ingredients generated the most waste and why
  - compare average order quantities against daily consumption rates
  - note whether demand spikes (weekends, late horizon) were handled correctly
  - record a short behavioral summary (2–4 bullet points) in the `notes` field of `results.tsv`
- use the behavioral inspection as the primary input for the next hypothesis, not just the score delta
- let the worker launcher commit, evaluate, and record the result
- expect the worker launcher to retry one failed evaluation before recording a crash
- do not change the evaluator, simulator, or frontier contract
