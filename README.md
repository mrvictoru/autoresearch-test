# autoresearch-test

Minimal framework inspired by [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), adapted for training task-specific models with a local-hosted LLM research agent.

Additional documentation:

- `docs/guide.md` for the full usage and architecture guide
- `docs/components.md` for component-level documentation
- `docs/mutation_guide.md` for mutation backend usage
- `docs/restaurant_benchmark.md` for the restaurant benchmark contract

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
- Mutation backend components:
  - `MutationRunner`, `MutationAgent`, `MutationProposal`
  - `Workspace` sandbox isolation with mutable-file whitelist
  - `SafeExecutor` timeout/log/failure classification
  - `research_brief.yaml` objective/constraint file

## Quick start

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run a small demo:

```bash
python -m autoresearch parametric
```

Run mutation mode with a research brief:

```bash
python -m autoresearch mutation \
  --task tiny_torch_classification \
  --brief research_brief.json \
  --agent-endpoint http://localhost:8080 \
  --agent-model local-model \
  --iterations 3
```

Run with trace and plots:

```bash
python -m autoresearch parametric --task blackjack --iterations 8 --trace --plot
```

Use a config file:

```bash
python -m autoresearch parametric --config experiment.json
```

## Harness-Driven Mode

Alongside `parametric` and `mutation` CLI modes, this repository also supports a Karpathy-style harness workflow where an external coding agent owns the optimization loop and this repo provides immutable benchmark infrastructure plus mutable experiment code.

Core harness files:

- `program.md`: run protocol, mutable/immutable boundaries, keep/discard rules
- `AGENTS.md`: agent-operating notes for harness sessions
- `results.tsv` / `mutation_results.tsv`: frontier ledgers used by harness runs
- `autoresearch/frontier.py`: git-frontier helpers for branch creation, commit/revert, and `results.tsv` bookkeeping
- `scripts/run_once.sh`: single-attempt harness helper for commit -> evaluate -> parse -> ledger update -> keep/discard

In this mode, treat evaluator and benchmark-defining files as immutable, and only mutate files allowed by the active experiment contract.

### Restaurant benchmark quick start

Run immutable restaurant evaluation against mutable train code:

```bash
python -m autoresearch.experiments.restaurant_eval \
  --experiment autoresearch/experiments/restaurant_train.py
```

Use the restaurant research brief for mutation runs:

```bash
python -m autoresearch mutation \
  --task restaurant_inventory \
  --brief research_brief_restaurant.json \
  --agent-endpoint http://localhost:8080 \
  --agent-model local-model \
  --iterations 5
```

Harness-driven single run:

```bash
./scripts/run_once.sh "adjust restaurant inventory policy"
```

That helper:

- commits tracked mutable changes,
- runs the immutable restaurant evaluator,
- parses `score` from stdout,
- appends the attempt to `results.tsv`,
- keeps only strict improvements over the current best kept run,
- reverts discarded or crashed candidate commits.

## Run with Docker

This repository can run entirely inside Docker, so you do not need Python installed locally.

Build the CUDA-based image:

```bash
docker compose build
```

Run the default demo in the container:

```bash
docker compose up autoresearch
```

Run tests in the container:

```bash
docker compose run --rm autoresearch python -m unittest discover -s tests -v
```

Open an interactive shell:

```bash
docker compose run --rm autoresearch bash
```

Start a Jupyter server in Docker:

```bash
docker compose up jupyter
```

The compose setup is tuned for a Linux/NVIDIA workstation:

- base image: `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04`
- host networking enabled so the container can reach a host-local LLM at `http://127.0.0.1:8080`
- `gpus: all` enabled for NVIDIA runtime access
- `ipc: host` enabled for larger ML workloads
- requires Docker Compose v1.28+ / Compose V2 and the NVIDIA Container Toolkit on the host

### Use your local LLM endpoint from Docker

By default, `compose.yaml` passes:

- `AUTORESEARCH_AGENT_ENDPOINT=http://127.0.0.1:8080`
- `AUTORESEARCH_AGENT_MODEL=local-model`

Override them in your shell or a `.env` file before running compose:

```bash
export AUTORESEARCH_AGENT_ENDPOINT=http://127.0.0.1:8080
export AUTORESEARCH_AGENT_MODEL=qwen2.5-coder
export AUTORESEARCH_TASK=restaurant_inventory
export AUTORESEARCH_ITERATIONS=10
export AUTORESEARCH_TRACE=1
docker compose up autoresearch
```

If you prefer a config file:

```bash
export AUTORESEARCH_CONFIG=experiment.json
docker compose up autoresearch
```

### Use Jupyter Notebook in Docker

The repo includes `notebooks/autoresearch_workflow.ipynb`.

Start the notebook server:

```bash
export JUPYTER_PORT=8888
export JUPYTER_TOKEN=autoresearch
docker compose up jupyter
```

Then open:

```text
http://127.0.0.1:8888/tree?token=autoresearch
```

The Jupyter container uses the same host networking setup, so notebooks can reach your host-local LLM endpoint at `http://127.0.0.1:8080`.

Use a local OpenAI-compatible endpoint:

```bash
python -m autoresearch parametric \
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
- `compose.yaml` uses Linux host networking so a host-local LLM endpoint remains reachable from the container.
- `docker compose up jupyter` starts a notebook server rooted at `/workspace`.
