from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .brief import ResearchBrief
from .core import ExperimentRecord, ExperimentStatus, MutationRunResult
from .executor import SafeExecutor
from .mutation_agent import MutationAgent
from .sandbox import Workspace


class MutationRunner:
    def __init__(
        self,
        *,
        agent: MutationAgent,
        executor: SafeExecutor,
    ) -> None:
        self.agent = agent
        self.executor = executor

    def run(
        self,
        *,
        task_name: str,
        source_root: str | Path,
        research_brief: ResearchBrief,
        iterations: int = 5,
    ) -> MutationRunResult:
        workspace = Workspace.create(source_root=source_root)
        history: list[ExperimentRecord] = []
        command = _resolve_experiment_command(research_brief)
        baseline_candidate = workspace.create_candidate()
        baseline_result = self.executor.run(
            command=command,
            cwd=baseline_candidate,
            log_dir=baseline_candidate / "artifacts" / "mutation" / "baseline",
        )
        if not baseline_result.success or baseline_result.score is None:
            raise RuntimeError("Baseline experiment failed; mutation loop cannot start")
        frontier_score = baseline_result.score
        workspace.promote_candidate(baseline_candidate)
        best_snapshot_id = workspace.next_snapshot_id(
            root=workspace.frontier_root,
            tracked_files=research_brief.allowed_mutable_files,
        )
        for iteration in range(1, iterations + 1):
            proposal = self.agent.propose_mutation(
                task_name=task_name,
                context=self._build_context(
                    workspace=workspace,
                    research_brief=research_brief,
                    frontier_score=frontier_score,
                    history=history,
                ),
            )
            candidate_root = workspace.create_candidate()
            iteration_dir = candidate_root / "artifacts" / "mutation" / f"iter-{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            auto_fixed = False
            try:
                workspace.apply_proposal(
                    candidate_root=candidate_root,
                    proposal=proposal,
                    allowed_mutable_files=research_brief.allowed_mutable_files,
                )
                result = self.executor.run(command=command, cwd=candidate_root, log_dir=iteration_dir)
                if (
                    not result.success
                    and result.failure_category == "syntax_error"
                    and not auto_fixed
                ):
                    retry_proposal = self.agent.propose_mutation(
                        task_name=task_name,
                        context=self._build_context(
                            workspace=workspace,
                            research_brief=research_brief,
                            frontier_score=frontier_score,
                            history=history,
                            last_error=Path(result.stderr_path).read_text(encoding="utf-8"),
                            retry_requested=True,
                        ),
                    )
                    workspace.apply_proposal(
                        candidate_root=candidate_root,
                        proposal=retry_proposal,
                        allowed_mutable_files=research_brief.allowed_mutable_files,
                    )
                    result = self.executor.run(
                        command=command,
                        cwd=candidate_root,
                        log_dir=iteration_dir / "retry",
                    )
                    auto_fixed = result.success
                snapshot_id = workspace.next_snapshot_id(
                    root=candidate_root,
                    tracked_files=research_brief.allowed_mutable_files,
                )
                if not result.success or result.score is None:
                    history.append(
                        ExperimentRecord(
                            iteration=iteration,
                            status=ExperimentStatus.CRASH,
                            description=proposal.description,
                            snapshot_id=snapshot_id,
                            score=None,
                            run_log_path=result.log_path,
                            resource_metrics={
                                "wall_time_seconds": result.wall_time_seconds,
                                "peak_memory_kb": result.peak_memory_kb,
                            },
                            failure_category=result.failure_category,
                            auto_fixed=auto_fixed,
                        )
                    )
                    workspace.discard_candidate(candidate_root)
                    continue
                if result.score > frontier_score:
                    frontier_score = result.score
                    workspace.promote_candidate(candidate_root)
                    best_snapshot_id = snapshot_id
                    status = ExperimentStatus.KEEP
                else:
                    status = ExperimentStatus.DISCARD
                    workspace.discard_candidate(candidate_root)
                history.append(
                    ExperimentRecord(
                        iteration=iteration,
                        status=status,
                        description=proposal.description,
                        snapshot_id=snapshot_id,
                        score=result.score,
                        run_log_path=result.stdout_path,
                        resource_metrics={
                            "wall_time_seconds": result.wall_time_seconds,
                            "peak_memory_kb": result.peak_memory_kb,
                            **result.metrics,
                        },
                        auto_fixed=auto_fixed,
                    )
                )
            except Exception as exc:
                snapshot_id = workspace.next_snapshot_id(
                    root=workspace.frontier_root,
                    tracked_files=research_brief.allowed_mutable_files,
                )
                history.append(
                    ExperimentRecord(
                        iteration=iteration,
                        status=ExperimentStatus.CRASH,
                        description=proposal.description,
                        snapshot_id=snapshot_id,
                        score=None,
                        run_log_path=str(iteration_dir / "exception.log"),
                        resource_metrics={},
                        failure_category=type(exc).__name__,
                    )
                )
                (iteration_dir / "exception.log").write_text(str(exc), encoding="utf-8")
                workspace.discard_candidate(candidate_root)
        return MutationRunResult(
            task_name=task_name,
            best_score=frontier_score,
            best_snapshot_id=best_snapshot_id,
            history=history,
        )

    def _build_context(
        self,
        *,
        workspace: Workspace,
        research_brief: ResearchBrief,
        frontier_score: float,
        history: list[ExperimentRecord],
        last_error: str | None = None,
        retry_requested: bool = False,
    ) -> dict[str, Any]:
        mutable_contents: dict[str, str] = {}
        immutable_contents: dict[str, str] = {}
        for relative_path in research_brief.allowed_mutable_files:
            path = workspace.frontier_root / relative_path
            if path.exists():
                mutable_contents[relative_path] = path.read_text(encoding="utf-8")
        for relative_path in research_brief.immutable_files:
            path = workspace.frontier_root / relative_path
            if path.exists():
                immutable_contents[relative_path] = path.read_text(encoding="utf-8")
        recent_success = [
            {"iteration": item.iteration, "score": item.score, "description": item.description}
            for item in history
            if item.status == ExperimentStatus.KEEP
        ][-5:]
        context = {
            "research_brief": research_brief.to_context(),
            "allowed_mutable_files": research_brief.allowed_mutable_files,
            "immutable_files": research_brief.immutable_files,
            "mutable_file_contents": mutable_contents,
            "immutable_file_contents": immutable_contents,
            "frontier_score": frontier_score,
            "recent_successful_experiments": recent_success,
            "resource_constraints": research_brief.constraints,
            "last_error_log": last_error,
            "retry_requested": retry_requested,
        }
        return context


def _resolve_experiment_command(brief: ResearchBrief) -> list[str]:
    command = brief.constraints.get("experiment_command")
    if isinstance(command, list) and all(isinstance(v, str) for v in command):
        return [value if value != "{python}" else sys.executable for value in command]
    return [
        sys.executable,
        "-m",
        "autoresearch.experiments.neural_eval",
        "--experiment",
        "autoresearch/experiments/neural_train.py",
    ]
