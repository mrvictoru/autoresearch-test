from __future__ import annotations

import json
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
    """Extract `key` from `key:value` or `key=value` formats; return fallback otherwise."""
    try:
        parsed = json.loads(suggestion)
        if isinstance(parsed, dict) and key in parsed:
            return int(parsed[key])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    pattern = rf"{re.escape(key)}\s*[:=]\s*(\d+)"
    match = re.search(pattern, suggestion, re.IGNORECASE)
    if match:
        return int(match.group(1))

    quoted_pattern = rf"\"{re.escape(key)}\"\s*:\s*(\d+)"
    quoted_match = re.search(quoted_pattern, suggestion, re.IGNORECASE)
    if quoted_match:
        return int(quoted_match.group(1))
    return fallback


class InventoryPolicyTrainer(TaskTrainer):
    def train_step(
        self,
        *,
        model_state: dict[str, Any],
        suggestion: str,
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        reorder_point = _extract_int(suggestion, "reorder_point", model_state["reorder_point"])
        target_stock = _extract_int(suggestion, "target_stock", model_state["target_stock"])
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
        threshold = _extract_int(suggestion, "hit_threshold", model_state["hit_threshold"])
        threshold = max(12, min(20, threshold))
        return {"hit_threshold": threshold}


class HyperparameterTrainer(TaskTrainer):
    """Generic trainer for neural-style hyperparameter search loops."""

    def train_step(
        self,
        *,
        model_state: dict[str, Any],
        suggestion: str,
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(model_state)
        current_lr = float(updated.get("learning_rate", 1e-3))
        current_layers = int(updated.get("num_layers", 2))
        current_heads = int(updated.get("num_heads", 2))

        try:
            parsed = json.loads(suggestion)
            if isinstance(parsed, dict):
                if "learning_rate" in parsed:
                    updated["learning_rate"] = max(1e-6, float(parsed["learning_rate"]))
                if "num_layers" in parsed:
                    updated["num_layers"] = max(1, int(parsed["num_layers"]))
                if "num_heads" in parsed:
                    updated["num_heads"] = max(1, int(parsed["num_heads"]))
                return updated
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        updated["learning_rate"] = _extract_float(suggestion, "learning_rate", current_lr)
        updated["num_layers"] = max(1, _extract_int(suggestion, "num_layers", current_layers))
        updated["num_heads"] = max(1, _extract_int(suggestion, "num_heads", current_heads))
        return updated


def _extract_float(suggestion: str, key: str, fallback: float) -> float:
    pattern = rf"{re.escape(key)}\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, suggestion, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return fallback


@dataclass
class TrainerRegistry:
    _trainers: dict[str, TaskTrainer] = field(default_factory=dict)

    def register(self, task_name: str, trainer: TaskTrainer) -> None:
        self._trainers[task_name] = trainer

    def get(self, task_name: str) -> TaskTrainer:
        if task_name not in self._trainers:
            raise KeyError(f"No trainer registered for task '{task_name}'")
        return self._trainers[task_name]
