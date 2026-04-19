# autoresearch-test

Minimal framework inspired by [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), adapted for training task-specific models with a local-hosted LLM research agent.

## What is included

- A reusable autoresearch loop (`AutoresearchRunner`) that:
  - asks a research agent for the next experiment or policy update
  - applies that suggestion through a task-specific trainer
  - evaluates performance, stores per-iteration metrics, and tracks the best model state
- A local LLM agent interface (`LocalLLMResearchAgent`) for OpenAI-compatible local endpoints.
- Traceability helpers:
  - `TraceableAgent` for prompt/response/latency capture
  - `RunResult.to_trace_log(...)` for JSON trace export
- Visualisation helpers:
  - `plot_run_result(...)` for matplotlib plots
  - `save_html_report(...)` for self-contained HTML reports
- Result exporters:
  - `RunResult.to_csv(...)`
  - `RunResult.to_trace_log(...)`
- Task and trainer interfaces to support multiple domains.
- Two example tasks:
  - restaurant inventory management simulation
  - blackjack strategy optimization
- Optional neural-task scaffolding:
  - `NeuralTask`
  - `TinyTorchClassificationTask`
  - `HyperparameterTrainer`

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

Use a local OpenAI-compatible endpoint:

```bash
python -m autoresearch.demo \
  --task restaurant_inventory \
  --iterations 10 \
  --agent-endpoint http://localhost:8080 \
  --agent-model mistral-7b \
  --prompt-preset concise \
  --temperature 0.3 \
  --trace
```

Example config:

```json
{
  "task": "restaurant_inventory",
  "task_params": {
    "days": 30,
    "seed": 42
  },
  "agent": {
    "endpoint": "http://localhost:8080",
    "model": "mistral-7b",
    "system_prompt": "You are a research optimizer.",
    "temperature": 0.3,
    "prompt_preset": "concise"
  },
  "iterations": 10,
  "output_dir": "artifacts"
}
```

Demo outputs per-task artifacts under `artifacts/<task_name>/`:

- `results.csv` (iteration history for spreadsheet analysis)
- `trace.json` (prompt/response/state trace log)
- `report.html` (self-contained visual report)
- `manifest.json` (reproducibility metadata including git commit hash)

## Notes

- `--plot` requires `matplotlib`.
- YAML configs are supported when `PyYAML` is installed.
- Neural tasks are optional and only activate when `torch` is installed.
