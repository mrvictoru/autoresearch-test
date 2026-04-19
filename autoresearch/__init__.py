"""Autoresearch-style framework for task-specific model training."""

from .agent import (
    AgentTrace,
    FewShotResearchAgent,
    LocalLLMResearchAgent,
    PROMPT_TEMPLATE_PRESETS,
    ResearchAgent,
    StructuredOutputAgent,
    TraceableAgent,
)
from .core import AutoresearchRunner, IterationRecord, RunResult
from .neural import HAS_TORCH, NeuralTask, TinyTorchClassificationTask
from .tasks import BlackjackTask, ResearchTask, RestaurantInventoryTask
from .training import (
    BlackjackPolicyTrainer,
    HyperparameterTrainer,
    InventoryPolicyTrainer,
    TaskTrainer,
    TrainerRegistry,
)
from .visualise import plot_run_result, save_html_report

__all__ = [
    "AgentTrace",
    "AutoresearchRunner",
    "BlackjackPolicyTrainer",
    "BlackjackTask",
    "FewShotResearchAgent",
    "HAS_TORCH",
    "HyperparameterTrainer",
    "InventoryPolicyTrainer",
    "IterationRecord",
    "LocalLLMResearchAgent",
    "NeuralTask",
    "PROMPT_TEMPLATE_PRESETS",
    "ResearchAgent",
    "ResearchTask",
    "RestaurantInventoryTask",
    "RunResult",
    "StructuredOutputAgent",
    "TaskTrainer",
    "TinyTorchClassificationTask",
    "TraceableAgent",
    "TrainerRegistry",
    "plot_run_result",
    "save_html_report",
]
