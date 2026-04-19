from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class RunResult:
    task_name: str
    best_model_state: dict[str, Any]
    best_score: float
    history: list[IterationRecord]


class AutoresearchRunner:
    """Task-agnostic autoresearch loop."""

    def __init__(self, *, agent: ResearchAgent, registry: TrainerRegistry) -> None:
        self.agent = agent
        self.registry = registry

    def run(self, task: ResearchTask, *, iterations: int = 5) -> RunResult:
        trainer = self.registry.get(task.name)
        model_state = task.initial_model_state()
        best_model_state = dict(model_state)
        best_score = task.evaluate_model(model_state)
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
            score = task.evaluate_model(model_state)
            if score > best_score:
                best_score = score
                best_model_state = dict(model_state)
            history.append(
                IterationRecord(
                    iteration=i,
                    suggestion=suggestion,
                    model_state=dict(model_state),
                    score=score,
                )
            )
        return RunResult(
            task_name=task.name,
            best_model_state=best_model_state,
            best_score=best_score,
            history=history,
        )
