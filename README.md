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
