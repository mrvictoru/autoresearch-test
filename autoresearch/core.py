from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import ResearchAgent
from .tasks import ResearchTask
from .training import TrainerRegistry


@dataclass
class IterationRecord:
    iteration: int
    suggestion: str
    model_state: dict[str, Any]
    score: float
    metrics: dict[str, Any]
    prompt_sent: str | None = None
    raw_response: str | None = None
    latency_seconds: float | None = None


@dataclass
class RunResult:
    task_name: str
    best_model_state: dict[str, Any]
    best_score: float
    history: list[IterationRecord]

    def to_csv(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metric_keys: set[str] = set()
        model_state_keys: set[str] = set()
        for entry in self.history:
            metric_keys.update(entry.metrics.keys())
            model_state_keys.update(entry.model_state.keys())
        fieldnames = [
            "task_name",
            "iteration",
            "score",
            "suggestion",
            "latency_seconds",
        ] + [f"metric_{k}" for k in sorted(metric_keys)] + [
            f"model_{k}" for k in sorted(model_state_keys)
        ]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.history:
                row: dict[str, Any] = {
                    "task_name": self.task_name,
                    "iteration": entry.iteration,
                    "score": entry.score,
                    "suggestion": entry.suggestion,
                    "latency_seconds": entry.latency_seconds,
                }
                for key in metric_keys:
                    row[f"metric_{key}"] = entry.metrics.get(key)
                for key in model_state_keys:
                    row[f"model_{key}"] = entry.model_state.get(key)
                writer.writerow(row)

    def to_trace_log(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_name": self.task_name,
            "best_model_state": self.best_model_state,
            "best_score": self.best_score,
            "iterations": [
                {
                    "iteration": entry.iteration,
                    "suggestion": entry.suggestion,
                    "model_state": entry.model_state,
                    "metrics": entry.metrics,
                    "score": entry.score,
                    "prompt_sent": entry.prompt_sent,
                    "raw_response": entry.raw_response,
                    "latency_seconds": entry.latency_seconds,
                }
                for entry in self.history
            ],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class AutoresearchRunner:
    """Task-agnostic autoresearch loop."""

    def __init__(self, *, agent: ResearchAgent, registry: TrainerRegistry) -> None:
        self.agent = agent
        self.registry = registry

    def run(self, task: ResearchTask, *, iterations: int = 5) -> RunResult:
        trainer = self.registry.get(task.name)
        model_state = task.initial_model_state()
        best_model_state = dict(model_state)
        initial_metrics = task.evaluate_model_detailed(model_state)
        best_score = float(initial_metrics["score"])
        history: list[IterationRecord] = []
        for i in range(1, iterations + 1):
            context = {
                "task_context": task.describe_context(),
                "best_score": best_score,
                "last_model_state": model_state,
            }
            suggestion = self.agent.propose(
                task_name=task.name,
                model_state=model_state,
                context=context,
            )
            model_state = trainer.train_step(
                model_state=model_state,
                suggestion=suggestion,
                task_context=task.describe_context(),
            )
            metrics = task.evaluate_model_detailed(model_state)
            score = float(metrics["score"])
            if score > best_score:
                best_score = score
                best_model_state = dict(model_state)
            trace = self.agent.get_last_trace()
            history.append(
                IterationRecord(
                    iteration=i,
                    suggestion=suggestion,
                    model_state=dict(model_state),
                    score=score,
                    metrics=metrics,
                    prompt_sent=trace.user_prompt if trace else None,
                    raw_response=trace.raw_response if trace else None,
                    latency_seconds=trace.latency_seconds if trace else None,
                )
            )
        return RunResult(
            task_name=task.name,
            best_model_state=best_model_state,
            best_score=best_score,
            history=history,
        )
