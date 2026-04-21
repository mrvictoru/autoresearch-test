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
from .brief import ResearchBrief, load_research_brief
from .core import (
    AutoresearchRunner,
    ExperimentRecord,
    ExperimentStatus,
    IterationRecord,
    MutationRunResult,
    ParametricRunner,
    RunResult,
)
from .executor import ExecutionResult, SafeExecutor
from .frontier import (
    append_result,
    commit_before_run,
    create_research_branch,
    get_current_sha,
    init_results_tsv,
    read_best_result,
    revert_last_commit,
)
from .harness import EvaluationHarness
from .mutation_agent import FileEdit, LocalLLMMutationAgent, MutationAgent, MutationProposal
from .mutation_runner import MutationRunner
from .neural import HAS_TORCH, NeuralTask, TinyTorchClassificationTask
from .tasks import BlackjackTask, ResearchTask, RestaurantInventoryTask
from .sandbox import Workspace
from .training import (
    BlackjackPolicyTrainer,
    HyperparameterTrainer,
    InventoryPolicyTrainer,
    TaskTrainer,
    TrainerRegistry,
)
from .visualise import plot_mutation_run, plot_run_result, save_html_report

__all__ = [
    "AgentTrace",
    "AutoresearchRunner",
    "append_result",
    "EvaluationHarness",
    "BlackjackPolicyTrainer",
    "BlackjackTask",
    "commit_before_run",
    "create_research_branch",
    "ExecutionResult",
    "ExperimentRecord",
    "ExperimentStatus",
    "FileEdit",
    "FewShotResearchAgent",
    "HAS_TORCH",
    "HyperparameterTrainer",
    "InventoryPolicyTrainer",
    "IterationRecord",
    "get_current_sha",
    "init_results_tsv",
    "LocalLLMResearchAgent",
    "LocalLLMMutationAgent",
    "MutationAgent",
    "MutationProposal",
    "MutationRunResult",
    "MutationRunner",
    "NeuralTask",
    "ParametricRunner",
    "PROMPT_TEMPLATE_PRESETS",
    "ResearchBrief",
    "ResearchAgent",
    "ResearchTask",
    "RestaurantInventoryTask",
    "RunResult",
    "read_best_result",
    "revert_last_commit",
    "SafeExecutor",
    "StructuredOutputAgent",
    "TaskTrainer",
    "TinyTorchClassificationTask",
    "TraceableAgent",
    "TrainerRegistry",
    "Workspace",
    "load_research_brief",
    "plot_mutation_run",
    "plot_run_result",
    "save_html_report",
]
