# Agent Harness Notes

1. Start by reading `program.md`.
2. Follow mutable/immutable boundaries exactly.
3. For neural mutation mode, only edit:
   - `autoresearch/experiments/neural_train.py`
4. Do not edit evaluator or benchmark-defining files unless explicitly authorized.
5. Use Git history and experiment logs as the optimization frontier.
6. Keep changes small and reproducible; run baseline first.
