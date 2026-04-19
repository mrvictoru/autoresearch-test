# autoresearch-test

Minimal framework inspired by [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), adapted for training task-specific models with a local-hosted LLM research agent.

## What is included

- A reusable autoresearch loop (`AutoresearchRunner`) that:
  - asks a research agent for the next experiment or policy update
  - applies that suggestion through a task-specific trainer
  - evaluates performance and tracks the best model state
- A local LLM agent interface (`LocalLLMResearchAgent`) for OpenAI-compatible local endpoints.
- Task and trainer interfaces to support multiple domains.
- Two example tasks:
  - restaurant inventory management simulation
  - blackjack strategy optimization

## Quick start

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run a small demo:

```bash
python -m autoresearch.demo
```

Run with trace and plots:

```bash
python -m autoresearch.demo --task blackjack --iterations 8 --trace --plot
```

Use a config file:

```bash
python -m autoresearch.demo --config experiment.json
```

Demo outputs per-task artifacts under `artifacts/<task_name>/`:

- `results.csv` (iteration history for spreadsheet analysis)
- `trace.json` (prompt/response/state trace log)
- `report.html` (self-contained visual report)
- `manifest.json` (reproducibility metadata including git commit hash)
