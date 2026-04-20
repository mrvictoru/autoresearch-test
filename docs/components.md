# Autoresearch-Test — Component Reference

This document describes the main components in `autoresearch/` and how they fit together.

## Package Overview

```text
autoresearch/
├── __init__.py
├── agent.py
├── brief.py
├── core.py
├── demo.py
├── executor.py
├── harness.py
├── mutation_agent.py
├── mutation_runner.py
├── neural.py
├── sandbox.py
├── tasks.py
├── training.py
├── visualise.py
└── experiments/
```

---

## `__init__.py`

**Purpose**

- Defines the public package surface.
- Re-exports the main framework classes and helpers for convenient imports.

**Use when**

- You want consumers to import from `autoresearch` instead of individual modules.

**Key exports**

- Agents: `ResearchAgent`, `LocalLLMResearchAgent`, `FewShotResearchAgent`, `StructuredOutputAgent`, `TraceableAgent`, `AgentTrace`
- Core loop: `AutoresearchRunner`, `IterationRecord`, `RunResult`
- Tasks: `ResearchTask`, `RestaurantInventoryTask`, `BlackjackTask`, `NeuralTask`, `TinyTorchClassificationTask`
- Trainers: `TaskTrainer`, `InventoryPolicyTrainer`, `BlackjackPolicyTrainer`, `HyperparameterTrainer`, `TrainerRegistry`
- Visualisation: `plot_run_result`, `save_html_report`

---

## `agent.py`

**Purpose**

- Defines the agent abstraction and concrete agent implementations.
- Builds prompts, calls a local OpenAI-compatible endpoint, and captures trace data.

### `AgentTrace`

**Role**

- Stores per-call metadata for observability and debugging.

**Fields**

- `task_name`
- `system_prompt`
- `user_prompt`
- `raw_response`
- `suggestion`
- `latency_seconds`

### `ResearchAgent`

**Role**

- Base interface for anything that can propose the next experiment step.

**Main method**

- `propose(task_name, model_state, context) -> str`

**Extension point**

- Implement this interface to add custom agents.

### `LocalLLMResearchAgent`

**Role**

- Calls a local OpenAI-compatible chat endpoint and returns the model suggestion.

**Responsibilities**

- Format the user prompt from task state and context
- Build chat messages
- POST to `/v1/chat/completions`
- Parse the returned assistant message
- Store the latest trace

**Main configuration**

- `endpoint`
- `model`
- `system_prompt`
- `user_prompt_template`
- `temperature`
- `timeout_seconds`

**Related helper**

- `from_preset(...)` loads one of `PROMPT_TEMPLATE_PRESETS`

### `FewShotResearchAgent`

**Role**

- Extends `LocalLLMResearchAgent` with example user/assistant pairs before the real prompt.

**Use when**

- The local model needs stronger output shaping from examples.

### `StructuredOutputAgent`

**Role**

- Forces replies into a normalized JSON object string.

**Use when**

- Trainers should consume structured output more reliably than free text.

### `TraceableAgent`

**Role**

- Decorator around any `ResearchAgent`.
- Preserves a list of traces and exposes the latest one.

**Use when**

- You want prompt/response/latency history even for simple stub agents.

### Internal helper

- `_parse_json_from_text(...)` extracts a JSON object from plain text or fenced code blocks

---

## `core.py`

**Purpose**

- Implements the task-agnostic autoresearch loop and result containers.

### `IterationRecord`

**Role**

- Represents one iteration of a run.

**Stored data**

- iteration number
- suggestion text
- model state after training
- scalar score
- detailed metrics
- prompt/response/latency trace fields

### `RunResult`

**Role**

- Final output of a run.

**Stored data**

- `task_name`
- `best_model_state`
- `best_score`
- `history`

**Export helpers**

- `to_csv(path)`
- `to_trace_log(path)`

### `AutoresearchRunner`

**Role**

- Orchestrates the agent → trainer → evaluator loop.

**Dependencies**

- a `ResearchAgent`
- a `TrainerRegistry`
- a concrete `ResearchTask`

**Runtime flow**

1. Load the trainer for the task
2. Get the initial model state
3. Evaluate the initial state
4. Ask the agent for a suggestion
5. Apply the suggestion through the trainer
6. Re-evaluate the updated model state
7. Track best score and append an `IterationRecord`
8. Return a `RunResult`

### Mutation records/results

- `ExperimentStatus`: `keep`, `discard`, `crash`
- `ExperimentRecord`: per-attempt mutation lifecycle entry
- `MutationRunResult`: mutation run aggregate with `to_csv(...)` and `to_experiment_log(...)`

---

## `tasks.py`

**Purpose**

- Defines the task abstraction and example task implementations.

### `ResearchTask`

**Role**

- Base contract for optimization targets.

**Required members**

- `name`
- `describe_context()`
- `initial_model_state()`
- `evaluate_model(model_state)`

**Optional override**

- `evaluate_model_detailed(model_state)`

### `RestaurantInventoryTask`

**Role**

- Simulates a restaurant inventory policy over a fixed number of days.

**Optimized parameters**

- `reorder_point`
- `target_stock`

**Outputs**

- `score`
- `stockouts`
- `waste_units`
- `total_orders`

### `BlackjackTask`

**Role**

- Simulates a simplified blackjack policy.

**Optimized parameter**

- `hit_threshold`

**Outputs**

- `score`
- `wins`
- `losses`
- `draws`
- `win_rate`

---

## `training.py`

**Purpose**

- Converts agent suggestions into validated model-state updates.

### `TaskTrainer`

**Role**

- Base interface for task-specific update logic.

**Main method**

- `train_step(model_state, suggestion, task_context) -> dict[str, Any]`

### `InventoryPolicyTrainer`

**Role**

- Parses `reorder_point` and `target_stock` from the suggestion.
- Enforces valid minimum values and ordering constraints.

### `BlackjackPolicyTrainer`

**Role**

- Parses `hit_threshold` from the suggestion.
- Clamps it to the supported range `12..20`.

### `HyperparameterTrainer`

**Role**

- Generic trainer for neural-style hyperparameter search.

**Handled fields**

- `learning_rate`
- `num_layers`
- `num_heads`

### `TrainerRegistry`

**Role**

- Maps task names to trainer instances.

**Main operations**

- `register(task_name, trainer)`
- `get(task_name)`

### Internal helpers

- `_extract_int(...)` parses integer values from JSON-like or key/value text
- `_extract_float(...)` parses float values from key/value text

---

## `visualise.py`

**Purpose**

- Produces notebook-friendly and standalone run visualizations.

### `plot_run_result`

**Role**

- Creates a matplotlib figure showing score history and optional parameter trajectories.

**Use when**

- Working interactively in Python or Jupyter.

### `save_html_report`

**Role**

- Writes a self-contained HTML report with a score chart and iteration table.

**Use when**

- You want a shareable artifact without requiring Python tooling to inspect it.

### `plot_mutation_run`

**Role**

- Plots mutation lifecycle: score trajectory, keep/discard/crash status, cumulative resource usage.

---

## `brief.py`

**Purpose**

- Defines `ResearchBrief` schema and loader for JSON/YAML brief files.

**Key API**

- `ResearchBrief.to_context()`
- `load_research_brief(path)`

---

## `harness.py`

**Purpose**

- Wraps task evaluator calls behind `EvaluationHarness` to enforce optional timeouts and score presence.

---

## `mutation_agent.py`

**Purpose**

- Defines structured mutation proposal interfaces.

**Key types**

- `FileEdit`
- `MutationProposal`
- `MutationAgent`
- `LocalLLMMutationAgent`

---

## `sandbox.py`

**Purpose**

- Provides `Workspace` for isolated frontier/candidate directories and file whitelist enforcement.

---

## `executor.py`

**Purpose**

- Provides `SafeExecutor` and `ExecutionResult` for guarded experiment execution and failure classification.

---

## `mutation_runner.py`

**Purpose**

- Orchestrates mutation lifecycle: baseline -> propose -> apply -> run -> keep/discard/crash.

---

## `experiments/neural_eval.py` and `experiments/neural_train.py`

**Purpose**

- Split immutable evaluator (`neural_eval.py`) from mutable training logic (`neural_train.py`) for mutation-safe neural experimentation.

---

## `neural.py`

**Purpose**

- Provides optional torch-backed experimentation scaffolding.

### `HAS_TORCH`

**Role**

- Boolean feature flag indicating whether PyTorch is available.

### `NeuralTask`

**Role**

- Base class for neural tasks.
- Adds `require_torch()` so optional dependencies fail with a clear runtime error.

### `TinyTorchClassificationTask`

**Role**

- Toy binary classification task for neural hyperparameter experiments.

**Model-state fields**

- `learning_rate`
- `num_layers`
- `num_heads`
- `hidden_dim`
- `checkpoint_path`

**Metrics**

- `score`
- `validation_accuracy`
- `validation_loss`

---

## `demo.py`

**Purpose**

- Provides the CLI entrypoint and end-to-end demo workflow.

### `CyclicDemoAgent`

**Role**

- Deterministic fallback agent for local testing without an LLM server.

### Main helpers

- `_build_registry()` creates the built-in task/trainer registry
- `_parse_args()` reads CLI flags
- `_load_config()` loads JSON or YAML experiment config
- `_build_agent()` selects local LLM vs demo agent
- `_build_tasks()` selects task instances from args/config
- `_write_manifest()` writes reproducibility metadata
- `_print_trace()` prints full per-iteration trace output
- `main()` runs the complete CLI flow

### CLI responsibilities

- parse arguments
- load config overrides
- initialize registry, agent, and runner
- run one or more tasks
- write CSV, trace, HTML report, and manifest artifacts
- optionally print traces and show plots

---

## Component Relationships

```text
ResearchAgent -> proposes suggestion text
TaskTrainer   -> turns suggestion into next model state
ResearchTask  -> evaluates model state and returns score/metrics
AutoresearchRunner -> coordinates the loop and records history
RunResult     -> exports artifacts and feeds visualization helpers
demo.py       -> wires all components together for CLI/container/notebook use
```

## Recommended Reading Order

1. `core.py`
2. `tasks.py`
3. `training.py`
4. `agent.py`
5. `visualise.py`
6. `demo.py`
7. `neural.py`
