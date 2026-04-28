# Experiment worker

Operate inside one isolated git worktree.

Rules:
- edit only `autoresearch/experiments/restaurant_train.py`
- implement exactly one hypothesis per run
- let the worker launcher commit, evaluate, and record the result
- expect the worker launcher to retry one failed evaluation before recording a crash
- do not change the evaluator, simulator, or frontier contract
