from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from .tasks import ResearchTask


class EvaluationHarness:
    """Read-only wrapper around task evaluation with optional timeout."""

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = timeout_seconds

    def evaluate_model_detailed(
        self, *, task: ResearchTask, model_state: dict[str, Any]
    ) -> dict[str, Any]:
        if self.timeout_seconds is None:
            metrics = task.evaluate_model_detailed(dict(model_state))
            if "score" not in metrics:
                raise ValueError("Task evaluator must return a 'score' field")
            return dict(metrics)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task.evaluate_model_detailed, dict(model_state))
            try:
                metrics = future.result(timeout=self.timeout_seconds)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Evaluation timed out after {self.timeout_seconds} seconds"
                ) from exc
        if "score" not in metrics:
            raise ValueError("Task evaluator must return a 'score' field")
        return dict(metrics)

    def evaluate_score_only(self, *, task: ResearchTask, model_state: dict[str, Any]) -> float:
        metrics = self.evaluate_model_detailed(task=task, model_state=model_state)
        return float(metrics["score"])
