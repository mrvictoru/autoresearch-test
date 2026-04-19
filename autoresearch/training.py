from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class TaskTrainer(ABC):
    """Updates model parameters for a given task based on an LLM suggestion."""

    @abstractmethod
    def train_step(
        self,
        *,
        model_state: dict[str, Any],
        suggestion: str,
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        pass


def _extract_int(suggestion: str, key: str, fallback: int) -> int:
    pattern = rf"{re.escape(key)}\s*[:=]\s*(-?\d+)"
    match = re.search(pattern, suggestion, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback


class InventoryPolicyTrainer(TaskTrainer):
    def train_step(
        self,
        *,
        model_state: dict[str, Any],
        suggestion: str,
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        reorder_point = _extract_int(
            suggestion, "reorder_point", int(model_state["reorder_point"])
        )
        target_stock = _extract_int(
            suggestion, "target_stock", int(model_state["target_stock"])
        )
        reorder_point = max(1, reorder_point)
        target_stock = max(reorder_point + 1, target_stock)
        return {"reorder_point": reorder_point, "target_stock": target_stock}


class BlackjackPolicyTrainer(TaskTrainer):
    def train_step(
        self,
        *,
        model_state: dict[str, Any],
        suggestion: str,
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        threshold = _extract_int(
            suggestion, "hit_threshold", int(model_state["hit_threshold"])
        )
        threshold = max(12, min(20, threshold))
        return {"hit_threshold": threshold}


@dataclass
class TrainerRegistry:
    _trainers: dict[str, TaskTrainer] = field(default_factory=dict)

    def register(self, task_name: str, trainer: TaskTrainer) -> None:
        self._trainers[task_name] = trainer

    def get(self, task_name: str) -> TaskTrainer:
        if task_name not in self._trainers:
            raise KeyError(f"No trainer registered for task '{task_name}'")
        return self._trainers[task_name]
