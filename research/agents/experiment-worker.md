# Experiment worker

Operate inside one isolated git worktree.

Rules:
- edit only `autoresearch/experiments/restaurant_train.py`
- implement exactly one hypothesis per run
- let the worker launcher commit, evaluate, and record the result
- do not change the evaluator, simulator, or frontier contract
