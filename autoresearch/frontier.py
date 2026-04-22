from __future__ import annotations

import csv
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_FIELDNAMES = ("timestamp", "branch", "sha", "score", "decision", "message")


def _git(repo_root: str | Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(repo_root).resolve()), *args],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip() or f"git {' '.join(args)} failed"
        raise RuntimeError(message)
    return completed.stdout.strip()


def _normalize_branch_tag(tag: str) -> str:
    """Normalize free-form tags for git branch names.

    Rules:
    - lower-case and trim surrounding whitespace
    - collapse one-or-more disallowed characters into a single dash
    - preserve allowed separators inside the tag: `.`, `_`, `-`
    - strip leading/trailing separators (`-`, `.`, `_`)
    """
    normalized = tag.strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("tag must contain at least one alphanumeric character")
    return normalized


def create_research_branch(tag: str, *, repo_root: str | Path = ".") -> str:
    branch_name = f"autoresearch/{_normalize_branch_tag(tag)}"
    _git(repo_root, "check-ref-format", "--branch", branch_name)
    _git(repo_root, "checkout", "-b", branch_name)
    return branch_name


def commit_before_run(message: str, *, repo_root: str | Path = ".") -> str:
    _git(repo_root, "add", "-u")
    _git(repo_root, "commit", "--allow-empty", "-m", message)
    return get_current_sha(repo_root=repo_root)


def revert_last_commit(*, repo_root: str | Path = ".") -> str:
    _git(repo_root, "reset", "--hard", "HEAD~1")
    return get_current_sha(repo_root=repo_root)


def get_current_sha(*, repo_root: str | Path = ".") -> str:
    return _git(repo_root, "rev-parse", "HEAD")


def init_results_tsv(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES, delimiter="\t")
        writer.writeheader()
    return target


def append_result(
    path: str | Path,
    *,
    branch: str,
    sha: str,
    score: float | None,
    decision: str,
    message: str = "",
) -> None:
    target = init_results_tsv(path)
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES, delimiter="\t")
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "branch": branch,
                "sha": sha,
                "score": "" if score is None else f"{score:.12g}",
                "decision": decision,
                "message": message,
            }
        )


def read_best_result(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    best_row: dict[str, Any] | None = None
    best_score: float | None = None
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("decision") != "keep":
                continue
            raw_score = row.get("score", "")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_row = dict(row)
                best_row["score"] = score
    return best_row
