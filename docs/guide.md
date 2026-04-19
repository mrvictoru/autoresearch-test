# Autoresearch-Test — Detailed Guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module-by-Module Walk-through](#module-by-module-walk-through)
4. [How a Run Executes End-to-End](#how-a-run-executes-end-to-end)
5. [Customisation Guide](#customisation-guide)
   - [Changing the LLM Research Agent Prompt](#changing-the-llm-research-agent-prompt)
   - [Adding a New Task](#adding-a-new-task)
   - [Switching the Experiment Model to an Attention-Based or Mamba-Based Architecture](#switching-the-experiment-model-to-an-attention-based-or-mamba-based-architecture)
6. [Running Tests](#running-tests)
7. [FAQ](#faq)

---

## Overview

This repository reimplements the core ideas of Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) project but generalises
them beyond the original setting.  Instead of training neural-network language
models, the framework lets you plug in **any task** (a restaurant inventory
simulator, a blackjack strategy optimiser, or anything else) and use a
**locally-hosted LLM** as the research agent that proposes what to try next.

The feedback loop is:

```
┌──────────────┐   suggestion   ┌───────────┐   new state   ┌──────────────┐
│ Research     │ ────────────►  │  Trainer   │ ───────────►  │  Task        │
│ Agent (LLM)  │                │            │               │  Evaluator   │
└──────┬───────┘                └────────────┘               └──────┬───────┘
       ▲                                                           │
       │              score + context                              │
       └───────────────────────────────────────────────────────────┘
```

Each loop iteration:

1. The **agent** looks at the current model state and context, then proposes
   updated parameters.
2. The **trainer** parses that proposal and produces a new model state.
3. The **task evaluator** scores the new state.
4. If the score beats the previous best, the state is saved.

---

## Architecture

```
autoresearch/
├── __init__.py        # Public API surface
├── agent.py           # ResearchAgent ABC + LocalLLMResearchAgent
├── core.py            # AutoresearchRunner, IterationRecord, RunResult
├── demo.py            # Self-contained demo with a stub agent
├── tasks.py           # ResearchTask ABC + two example tasks
└── training.py        # TaskTrainer ABC + trainers + TrainerRegistry

tests/
└── test_autoresearch.py   # Unit tests
```

The framework is deliberately **pure-Python** with zero external dependencies,
so it can run anywhere Python 3.10+ is available.

---

## Module-by-Module Walk-through

### `agent.py` — Research Agents

| Class | Purpose |
|---|---|
| `ResearchAgent` (ABC) | Defines the single method `propose(task_name, model_state, context) → str`. |
| `LocalLLMResearchAgent` | Concrete implementation that calls a local OpenAI-compatible `/v1/chat/completions` endpoint. |

`LocalLLMResearchAgent` accepts:

- `endpoint` — base URL of the local LLM server (e.g. `http://localhost:8080`).
- `model` — model identifier expected by that server.
- `system_prompt` — the system message sent to the LLM.
- `timeout_seconds` — HTTP timeout.

### `tasks.py` — Task Definitions

| Class | Purpose |
|---|---|
| `ResearchTask` (ABC) | Requires `name`, `describe_context()`, `initial_model_state()`, `evaluate_model()`. |
| `RestaurantInventoryTask` | Simulates daily demand, tracks stockouts & waste, returns negative total cost. |
| `BlackjackTask` | Plays simplified blackjack rounds and returns mean reward. |

The `evaluate_model()` method always returns a **scalar score where higher is
better**.

### `training.py` — Trainers & Registry

| Class / Function | Purpose |
|---|---|
| `TaskTrainer` (ABC) | `train_step(model_state, suggestion, task_context) → new_model_state` |
| `_extract_int(suggestion, key, fallback)` | Regex helper that pulls `key=value` or `key:value` from the LLM response. |
| `InventoryPolicyTrainer` | Extracts `reorder_point` and `target_stock` from a suggestion string. |
| `BlackjackPolicyTrainer` | Extracts `hit_threshold`, clamped to `[12, 20]`. |
| `TrainerRegistry` | Maps task names → trainers so the runner can look up the right one. |

### `core.py` — The Runner

`AutoresearchRunner.run(task, iterations)` drives the loop described in the
overview.  It returns a `RunResult` containing the best model state, its score,
and the full iteration history.

### `demo.py` — Quick Demo

Contains a `CyclicDemoAgent` that returns hard-coded suggestions (no LLM
required) and a `main()` function that runs both tasks for four iterations each.

---

## How a Run Executes End-to-End

```python
from autoresearch import (
    AutoresearchRunner, TrainerRegistry,
    RestaurantInventoryTask, InventoryPolicyTrainer,
)
from autoresearch.agent import LocalLLMResearchAgent

# 1. Set up the research agent
agent = LocalLLMResearchAgent(
    endpoint="http://localhost:8080",
    model="mistral-7b",
)

# 2. Register trainers
registry = TrainerRegistry()
registry.register("restaurant_inventory", InventoryPolicyTrainer())

# 3. Create the runner and launch
runner = AutoresearchRunner(agent=agent, registry=registry)
result = runner.run(RestaurantInventoryTask(), iterations=10)

print(result.best_model_state, result.best_score)
```

Behind the scenes the runner calls `agent.propose()`, which sends the current
state to the local LLM.  The LLM replies with something like
`reorder_point=24 target_stock=55`.  The `InventoryPolicyTrainer` parses that
string, validates the values, and returns an updated state dict.  The task
evaluator simulates 14 days of demand and returns the negative total cost.

---

## Customisation Guide

### Changing the LLM Research Agent Prompt

The prompt sent to the LLM consists of two parts:

1. **System prompt** — set via the `system_prompt` constructor argument of
   `LocalLLMResearchAgent`.
2. **User prompt** — assembled automatically inside `propose()`.

To change the system prompt:

```python
agent = LocalLLMResearchAgent(
    endpoint="http://localhost:8080",
    model="mistral-7b",
    system_prompt=(
        "You are an expert operations researcher. "
        "Given the current inventory policy and performance, "
        "suggest improved reorder_point and target_stock values. "
        "Reply ONLY with key=value pairs, one per line."
    ),
)
```

To change the **user prompt template**, subclass `LocalLLMResearchAgent` and
override `propose()`:

```python
class CustomAgent(LocalLLMResearchAgent):
    def propose(self, *, task_name, model_state, context):
        # Build your own prompt however you like, then call the LLM
        user_prompt = f"Optimise the {task_name} policy. State: {model_state}"
        # You can reuse the HTTP helper from the parent or call self directly.
        ...
```

### Adding a New Task

You need two things: a **task** and a **trainer**.

#### Step 1 — Create the task

Create a new file or add a class to `tasks.py`:

```python
from autoresearch.tasks import ResearchTask

class TicTacToeTask(ResearchTask):
    @property
    def name(self) -> str:
        return "tic_tac_toe"

    def describe_context(self) -> dict:
        return {"board_size": 3, "goal": "maximise win rate"}

    def initial_model_state(self) -> dict:
        # Whatever parameters your model/strategy exposes
        return {"aggression": 5, "defence": 5}

    def evaluate_model(self, model_state: dict) -> float:
        # Run simulated games and return a score (higher = better)
        ...
        return win_rate
```

#### Step 2 — Create the trainer

```python
from autoresearch.training import TaskTrainer, _extract_int

class TicTacToeTrainer(TaskTrainer):
    def train_step(self, *, model_state, suggestion, task_context):
        aggression = _extract_int(suggestion, "aggression", model_state["aggression"])
        defence = _extract_int(suggestion, "defence", model_state["defence"])
        return {"aggression": max(0, min(10, aggression)),
                "defence": max(0, min(10, defence))}
```

#### Step 3 — Register and run

```python
registry.register("tic_tac_toe", TicTacToeTrainer())
result = runner.run(TicTacToeTask(), iterations=10)
```

### Switching the Experiment Model to an Attention-Based or Mamba-Based Architecture

The current tasks use simple parameter dicts as the "model state".  To use a
real neural network (e.g. a Transformer or Mamba model) you would:

1. **Extend `model_state`** to hold serialised model weights (or a path to a
   checkpoint file).

2. **Update the task evaluator** to load and run the neural network:

   ```python
   def evaluate_model(self, model_state: dict) -> float:
       model = load_checkpoint(model_state["checkpoint_path"])
       # run inference / evaluation
       return score
   ```

3. **Update the trainer** to apply the LLM suggestion as a hyperparameter
   change, then actually train the model:

   ```python
   class AttentionModelTrainer(TaskTrainer):
       def train_step(self, *, model_state, suggestion, task_context):
           lr = _extract_float(suggestion, "learning_rate", 1e-4)
           epochs = _extract_int(suggestion, "epochs", 5)
           model = load_checkpoint(model_state["checkpoint_path"])
           train(model, lr=lr, epochs=epochs)
           new_path = save_checkpoint(model)
           return {"checkpoint_path": new_path, "lr": lr, "epochs": epochs}
   ```

4. **For Mamba-based models** the process is identical — only the model
   definition and training loop change.  The autoresearch framework does not
   care what kind of model you use; it only needs a `dict` that describes the
   current state and a `float` score from the evaluator.

> **Key insight:** the framework treats the model as a black box.  You can swap
> the entire model architecture without changing `AutoresearchRunner` or
> `ResearchAgent`.

---

## Running Tests

```bash
# Run the full test suite
python -m unittest discover -s tests -v

# Run the demo (no LLM server required)
python -m autoresearch.demo
```

---

## FAQ

**Q: Do I need a GPU to run this?**
No.  The framework itself is pure Python.  If your task trains a neural
network, GPU acceleration is helpful but optional.

**Q: Which local LLM servers are compatible?**
Any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
Popular choices include [llama.cpp](https://github.com/ggerganov/llama.cpp),
[Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm),
and [LM Studio](https://lmstudio.ai).

**Q: Can I run the framework without a local LLM?**
Yes — use the demo (`python -m autoresearch.demo`) or write a custom
`ResearchAgent` subclass that generates suggestions with any strategy you like
(random search, Bayesian optimisation, hard-coded rules, etc.).

**Q: How do I add more extraction helpers (e.g. `_extract_float`)?**
Add them to `training.py` following the same pattern as `_extract_int`.  Use
a regex like `r"key\s*[:=]\s*([0-9.]+)"` and `float()` instead of `int()`.
