"""Autoresearch-style framework for task-specific model training."""

from .agent import LocalLLMResearchAgent, ResearchAgent
from .core import AutoresearchRunner, IterationRecord, RunResult
from .tasks import BlackjackTask, ResearchTask, RestaurantInventoryTask
from .training import (
    BlackjackPolicyTrainer,
    InventoryPolicyTrainer,
    TaskTrainer,
    TrainerRegistry,
)

__all__ = [
    "AutoresearchRunner",
    "BlackjackPolicyTrainer",
    "BlackjackTask",
    "InventoryPolicyTrainer",
    "IterationRecord",
    "LocalLLMResearchAgent",
    "ResearchAgent",
    "ResearchTask",
    "RestaurantInventoryTask",
    "RunResult",
    "TaskTrainer",
    "TrainerRegistry",
]
