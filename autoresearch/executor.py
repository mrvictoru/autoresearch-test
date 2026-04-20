from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    score: float | None
    metrics: dict[str, Any]
    stdout_path: str
    stderr_path: str
    wall_time_seconds: float
    peak_memory_kb: int | None
    failure_category: str | None

    @property
    def log_path(self) -> str:
        return self.stderr_path if self.failure_category else self.stdout_path


class SafeExecutor:
    def __init__(
        self,
        *,
        timeout_seconds: int = 60,
        cpu_time_limit_seconds: int | None = None,
        memory_limit_mb: int | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.cpu_time_limit_seconds = cpu_time_limit_seconds
        self.memory_limit_mb = memory_limit_mb

    def run(
        self,
        *,
        command: list[str],
        cwd: str | Path,
        log_dir: str | Path,
        metrics_path: str | Path | None = None,
    ) -> ExecutionResult:
        target_dir = Path(log_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = target_dir / "stdout.log"
        stderr_path = target_dir / "stderr.log"
        before_rusage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=Path(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                preexec_fn=self._build_preexec_fn(),
            )
            wall_time = time.perf_counter() - start
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            after_rusage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peak_memory_kb = max(0, int(after_rusage - before_rusage))
            if proc.returncode != 0:
                failure = self._classify_failure(stderr=proc.stderr, stdout=proc.stdout)
                return ExecutionResult(
                    success=False,
                    score=None,
                    metrics={},
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    wall_time_seconds=wall_time,
                    peak_memory_kb=peak_memory_kb,
                    failure_category=failure,
                )
            metrics = self._extract_metrics(proc.stdout, cwd=Path(cwd), metrics_path=metrics_path)
            return ExecutionResult(
                success=True,
                score=float(metrics.get("score")) if "score" in metrics else None,
                metrics=metrics,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                wall_time_seconds=wall_time,
                peak_memory_kb=peak_memory_kb,
                failure_category=None,
            )
        except subprocess.TimeoutExpired as exc:
            wall_time = time.perf_counter() - start
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "Execution timed out", encoding="utf-8")
            return ExecutionResult(
                success=False,
                score=None,
                metrics={},
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                wall_time_seconds=wall_time,
                peak_memory_kb=None,
                failure_category="timeout",
            )

    def _build_preexec_fn(self):
        if self.cpu_time_limit_seconds is None and self.memory_limit_mb is None:
            return None

        def _limit() -> None:
            if self.cpu_time_limit_seconds is not None:
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.cpu_time_limit_seconds, self.cpu_time_limit_seconds),
                )
            if self.memory_limit_mb is not None:
                limit_bytes = self.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

        return _limit

    @staticmethod
    def _extract_metrics(
        stdout: str, *, cwd: Path, metrics_path: str | Path | None
    ) -> dict[str, Any]:
        if metrics_path:
            path = cwd / Path(metrics_path)
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        marker = "METRIC_JSON:"
        for line in reversed(stdout.splitlines()):
            if marker in line:
                payload = line.split(marker, 1)[1].strip()
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise ValueError("METRIC_JSON payload must be an object")
                return parsed
        return {}

    @staticmethod
    def _classify_failure(*, stderr: str, stdout: str) -> str:
        text = f"{stderr}\n{stdout}".lower()
        if "syntaxerror" in text:
            return "syntax_error"
        if "importerror" in text or "modulenotfounderror" in text:
            return "import_error"
        if "out of memory" in text or "cuda out of memory" in text or "memoryerror" in text:
            return "oom"
        if "killed" in text and os.name != "nt":
            return "oom"
        return "runtime_error"
