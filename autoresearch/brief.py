from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResearchBrief:
    goal: str
    constraints: dict[str, Any]
    allowed_mutable_files: list[str]
    immutable_files: list[str]
    time_budget_seconds: int
    tie_breaker_policy: str

    def to_context(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "constraints": self.constraints,
            "allowed_mutable_files": self.allowed_mutable_files,
            "immutable_files": self.immutable_files,
            "time_budget_seconds": self.time_budget_seconds,
            "tie_breaker_policy": self.tie_breaker_policy,
        }


def load_research_brief(path: str | Path) -> ResearchBrief:
    source = Path(path)
    content = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(content)
    elif source.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML research brief requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(content) or {}
    else:
        raise ValueError("Unsupported research brief format. Use .json, .yml, or .yaml")
    required = [
        "goal",
        "constraints",
        "allowed_mutable_files",
        "immutable_files",
        "time_budget_seconds",
        "tie_breaker_policy",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Research brief is missing required fields: {', '.join(missing)}")
    return ResearchBrief(
        goal=str(raw["goal"]),
        constraints=dict(raw["constraints"]),
        allowed_mutable_files=list(raw["allowed_mutable_files"]),
        immutable_files=list(raw["immutable_files"]),
        time_budget_seconds=int(raw["time_budget_seconds"]),
        tie_breaker_policy=str(raw["tie_breaker_policy"]),
    )
