from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _normalize_branch_tag(tag: str) -> str:
    normalized = tag.strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("tag must contain at least one alphanumeric character")
    return normalized


def create_research_branch(tag: str, *, repo_root: str | Path = ".") -> str:
    branch_name = f"autoresearch/{_normalize_branch_tag(tag)}"
    completed = subprocess.run(
        ["git", "-C", str(Path(repo_root).resolve()), "checkout", "-b", branch_name],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip() or "git checkout failed"
        raise RuntimeError(message)
    return branch_name
