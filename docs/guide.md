# Autoresearch-Test — Detailed Guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Workflow](#core-workflow)
4. [Observability and Exports](#observability-and-exports)
5. [Agent Customisation](#agent-customisation)
6. [CLI and Config Files](#cli-and-config-files)
7. [Mutation Backend](#mutation-backend)
8. [Research Brief](#research-brief)
9. [Component Reference](#component-reference)
10. [Adding a New Task](#adding-a-new-task)
11. [Optional Neural-Task Support](#optional-neural-task-support)
12. [Docker and Docker Compose](#docker-and-docker-compose)
13. [Running Tests](#running-tests)
14. [FAQ](#faq)

---

## Overview

`autoresearch-test` is a small experimentation framework inspired by
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch). A run loops through:

1. a **research agent** that proposes the next experiment,
2. a **trainer** that applies that suggestion to a model state,
3. a **task evaluator** that returns a score and optional detailed metrics.

The framework now includes:

- iteration tracing (prompt, raw response, latency),
- detailed per-iteration metrics,
- CSV and JSON trace export,
- matplotlib and HTML reporting helpers,
- configurable prompt templates and temperature,
- JSON/YAML demo configuration,
- optional torch-gated neural-task scaffolding.

---

## Architecture

```text
autoresearch/
├── __init__.py        # Public API exports
├── agent.py           # ResearchAgent implementations, traces, prompt presets
├── brief.py           # Research brief schema loader
├── core.py            # AutoresearchRunner, IterationRecord, RunResult
├── demo.py            # CLI demo/config entrypoint
├── executor.py        # Safe subprocess execution with timeout/resource constraints
├── harness.py         # Immutable evaluation wrapper for task scoring
├── mutation_agent.py  # MutationAgent + structured MutationProposal
├── mutation_runner.py # Mutation lifecycle orchestration (keep/discard/crash)
├── neural.py          # Optional torch-gated neural task scaffold
├── sandbox.py         # Isolated workspace and mutable-file whitelist enforcement
├── tasks.py           # ResearchTask ABC + example tasks
├── training.py        # Trainers, parsing helpers, registry
├── visualise.py       # Plotting and HTML report helpers
└── experiments/       # Mutation-safe neural experiment split
   ├── neural_eval.py  # Immutable evaluator
   └── neural_train.py # Mutable training implementation

tests/
└── test_autoresearch.py
```

The base framework remains lightweight. Optional features require extra packages:

- `matplotlib` for `--plot` / `plot_run_result(...)`
- `PyYAML` for `.yaml` / `.yml` config files
- `torch` for neural-task examples

---

## Core Workflow

### Agents

`agent.py` exposes:

- `ResearchAgent` — abstract base class
- `LocalLLMResearchAgent` — local OpenAI-compatible client
- `TraceableAgent` — wrapper that records trace metadata
- `FewShotResearchAgent` — injects example state/suggestion pairs
- `StructuredOutputAgent` — normalises replies into JSON
- `PROMPT_TEMPLATE_PRESETS` — built-in prompt templates

`LocalLLMResearchAgent` now supports:

- `system_prompt`
- `user_prompt_template`
- `temperature`
- `timeout_seconds`

### Tasks

`ResearchTask` defines:

- `name`
- `describe_context()`
- `initial_model_state()`
- `evaluate_model()`
- `evaluate_model_detailed()` — defaults to `{"score": evaluate_model(...)}`

Built-in example tasks now emit richer metrics:

- `RestaurantInventoryTask`: `score`, `stockouts`, `waste_units`, `total_orders`
- `BlackjackTask`: `score`, `wins`, `losses`, `draws`, `win_rate`

### Runner and Results

`AutoresearchRunner.run(...)` returns a `RunResult` with:

- `best_model_state`
- `best_score`
- `history`

Each `IterationRecord` now stores:

- `suggestion`
- `model_state`
- `score`
- `metrics`
- `prompt_sent`
- `raw_response`
- `latency_seconds`

---

## Observability and Exports

### Tracing

Wrap any agent with `TraceableAgent` to preserve prompt/response/latency data:

```python
from autoresearch import TraceableAgent, LocalLLMResearchAgent

agent = TraceableAgent(
    LocalLLMResearchAgent(
        endpoint="http://localhost:8080",
        model="mistral-7b",
        temperature=0.3,
    )
)
```

### Export helpers

`RunResult` includes:

- `to_csv(path)` — iteration table for spreadsheets or pandas
- `to_trace_log(path)` — JSON trace dump for postmortem inspection

### Visualisation

`visualise.py` provides:

- `plot_run_result(result, include_parameters=True, show=True)`
- `save_html_report(result, path)`

Example:

```python
from autoresearch import plot_run_result, save_html_report

plot_run_result(result)
save_html_report(result, "artifacts/restaurant_inventory/report.html")
```

---

## Agent Customisation

### Prompt templates

Built-in presets:

- `concise`
- `chain-of-thought`
- `json-only`

Example:

```python
from autoresearch import LocalLLMResearchAgent

agent = LocalLLMResearchAgent.from_preset(
    endpoint="http://localhost:8080",
    model="mistral-7b",
    prompt_preset="concise",
    temperature=0.2,
)
```

You can also pass your own template via `user_prompt_template`. Supported placeholders include:

- `{task_name}`
- `{model_state}`
- `{context}`
- `{model_state_json}`
- `{context_json}`

### Few-shot prompting

```python
from autoresearch import FewShotResearchAgent

agent = FewShotResearchAgent(
    endpoint="http://localhost:8080",
    model="mistral-7b",
    few_shot_examples=[
        ({"hit_threshold": 16}, '{"hit_threshold": 17}')
    ],
)
```

### Structured JSON output

```python
from autoresearch import StructuredOutputAgent

agent = StructuredOutputAgent(
    endpoint="http://localhost:8080",
    model="mistral-7b",
)
```

This is useful when trainers prefer parsing JSON over regex-based extraction.

---

## Component Reference

For file-by-file component documentation, see:

- `docs/components.md`
- `docs/mutation_guide.md`

This companion document describes each package module, its primary classes/functions, and how the components interact.

---

## CLI and Config Files

The demo is now a small CLI:

```bash
python -m autoresearch.demo --task blackjack --iterations 8 --trace --plot
```

Supported flags include:

- `--config`
- `--mode`
- `--task`
- `--iterations`
- `--plot`
- `--trace`
- `--output-dir`
- `--agent-endpoint`
- `--agent-model`
- `--prompt-preset`
- `--temperature`
- `--brief`

---

## Mutation Backend

Mutation mode adds code-edit experiments while keeping the parameter loop as default:

- `ParametricRunner` / `AutoresearchRunner`: existing parameter-update loop
- `MutationRunner`: proposal -> sandbox apply -> safe execute -> keep/discard/crash
- `MutationAgent`: structured edit proposals (`patch` or `edits`) with file targets
- `Workspace`: isolated mutable frontier in temporary workspace
- `SafeExecutor`: timeout, log capture, metric extraction, failure categorization

---

## Research Brief

Mutation runs consume a research brief (`research_brief.yaml` or `research_brief.json` in repo root) with:

- `goal`
- `constraints`
- `allowed_mutable_files`
- `immutable_files`
- `time_budget_seconds`
- `tie_breaker_policy`

This file is loaded by `load_research_brief(...)` and passed into runner/agent context.

### JSON config example

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
  "iterations": 20,
  "output_dir": "artifacts"
}
```

### YAML config example

```yaml
task: restaurant_inventory
task_params:
  days: 30
  seed: 42
agent:
  endpoint: http://localhost:8080
  model: mistral-7b
  system_prompt: You are a research optimizer.
  temperature: 0.3
  prompt_preset: concise
iterations: 20
output_dir: artifacts
```

Each task writes artifacts under `artifacts/<task_name>/`:

- `results.csv`
- `trace.json`
- `report.html`
- `manifest.json`

The manifest stores the resolved config, timestamp, task name, iteration count, best score, and git commit hash.

---

## Adding a New Task

You usually need:

1. a `ResearchTask` implementation,
2. a `TaskTrainer` implementation,
3. a registry entry.

Example:

```python
from autoresearch.tasks import ResearchTask
from autoresearch.training import TaskTrainer, _extract_int

class TicTacToeTask(ResearchTask):
    @property
    def name(self) -> str:
        return "tic_tac_toe"

    def describe_context(self) -> dict:
        return {"board_size": 3, "goal": "maximize win rate"}

    def initial_model_state(self) -> dict:
        return {"aggression": 5, "defence": 5}

    def evaluate_model_detailed(self, model_state: dict) -> dict:
        win_rate = 0.6
        return {"score": win_rate, "win_rate": win_rate}

class TicTacToeTrainer(TaskTrainer):
    def train_step(self, *, model_state, suggestion, task_context):
        aggression = _extract_int(suggestion, "aggression", model_state["aggression"])
        defence = _extract_int(suggestion, "defence", model_state["defence"])
        return {"aggression": aggression, "defence": defence}
```

Then register and run it:

```python
registry.register("tic_tac_toe", TicTacToeTrainer())
result = runner.run(TicTacToeTask(), iterations=10)
```

---

## Optional Neural-Task Support

`neural.py` adds a torch-gated scaffold:

- `NeuralTask`
- `TinyTorchClassificationTask`
- `HyperparameterTrainer`

This path is optional; the framework still works without torch installed.

The intended pattern is:

- model state stores hyperparameters and checkpoint information,
- trainer applies LLM-suggested hyperparameter changes,
- task evaluator runs training/evaluation and returns detailed metrics.

---

## Docker and Docker Compose

The repository now includes:

- `Dockerfile` based on `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04`
- `compose.yaml` for one-command local runs
- `docker/entrypoint.sh` to translate environment variables into CLI flags

This setup is aimed at Linux hosts with NVIDIA GPUs. The compose service enables:

- `network_mode: host`
- `gpus: all`
- `ipc: host`
- Docker Compose v1.28+ / Compose V2 and the NVIDIA Container Toolkit are required on the host

That means a host-local OpenAI-compatible LLM endpoint can be reached directly at:

```text
http://127.0.0.1:8080
```

### Build and run

```bash
docker compose build
docker compose up autoresearch
```

### Run tests in the container

```bash
docker compose run --rm autoresearch python -m unittest discover -s tests -v
```

### Open a shell in the container

```bash
docker compose run --rm autoresearch bash
```

### Start Jupyter Notebook in the container

```bash
export JUPYTER_PORT=8888
export JUPYTER_TOKEN=autoresearch
docker compose up jupyter
```

Then open:

```text
http://127.0.0.1:8888/tree?token=autoresearch
```

The notebook server is rooted at `/workspace`, so the example notebook and generated artifacts are visible from both the host and the container.

### Configure the containerized run

`compose.yaml` reads these environment variables:

- `AUTORESEARCH_CONFIG`
- `AUTORESEARCH_TASK`
- `AUTORESEARCH_ITERATIONS`
- `AUTORESEARCH_OUTPUT_DIR`
- `AUTORESEARCH_TRACE`
- `AUTORESEARCH_PLOT`
- `AUTORESEARCH_AGENT_ENDPOINT`
- `AUTORESEARCH_AGENT_MODEL`
- `AUTORESEARCH_PROMPT_PRESET`
- `AUTORESEARCH_TEMPERATURE`

Example:

```bash
export AUTORESEARCH_AGENT_ENDPOINT=http://127.0.0.1:8080
export AUTORESEARCH_AGENT_MODEL=qwen2.5-coder
export AUTORESEARCH_TASK=blackjack
export AUTORESEARCH_ITERATIONS=8
export AUTORESEARCH_TRACE=1
docker compose up autoresearch
```

If you prefer config-driven execution:

```bash
export AUTORESEARCH_CONFIG=experiment.json
docker compose up autoresearch
```

### Notebook workflow

An example notebook is included at:

```text
notebooks/autoresearch_workflow.ipynb
```

It:

- runs the existing autoresearch workflow,
- selects a local LLM agent when `AUTORESEARCH_AGENT_ENDPOINT` and `AUTORESEARCH_AGENT_MODEL` are set,
- otherwise falls back to the deterministic demo agent,
- visualizes score and parameter trajectories with matplotlib,
- exports CSV, trace, and HTML report artifacts under `artifacts/notebooks/`.

---

## Running Tests

```bash
python -m unittest discover -s tests -v
python -m autoresearch.demo
```

Optional examples:

```bash
python -m autoresearch.demo --task blackjack --trace
python -m autoresearch.demo --config experiment.json
```

---

## FAQ

**Do I need a GPU?**  
No. The base framework does not. Only optional neural-task experiments may benefit from one.

**Do I need a local LLM server to try the project?**  
No. `python -m autoresearch.demo` uses a deterministic stub agent by default.

**Which local LLM servers are compatible?**  
Any server exposing an OpenAI-compatible `/v1/chat/completions` endpoint.

**What do I use for spreadsheet analysis?**  
Export `RunResult.to_csv(...)` or use the demo-generated `results.csv`.

**How do I inspect what the agent saw and returned?**  
Use `TraceableAgent`, `--trace`, or `RunResult.to_trace_log(...)`.
