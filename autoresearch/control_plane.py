from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .frontier import append_result, commit_before_run, init_results_tsv, read_best_result, revert_last_commit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUTABLE_FILE = Path("autoresearch/experiments/restaurant_train.py")
DEFAULT_RESULTS_PATH = Path("results.tsv")
DEFAULT_STATE_DIR = Path("research/state")
DEFAULT_WORKERS_DIR = DEFAULT_STATE_DIR / "workers"
DEFAULT_WORKTREE_DIR = Path("artifacts/control-plane/worktrees")
DEFAULT_LOG_DIR = Path("artifacts/control-plane/logs")
DEFAULT_EVALUATION_COMMAND = [
    "docker",
    "compose",
    "run",
    "--rm",
    "autoresearch",
    "python",
    "-m",
    "autoresearch.experiments.restaurant_eval",
    "--experiment",
    str(DEFAULT_MUTABLE_FILE),
]
FINAL_WORKER_STATUSES = {"completed", "promoted", "cleaned"}


class ControlPlaneError(RuntimeError):
    """Raised when control-plane operations cannot be completed safely."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo_root: str | Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(repo_root).resolve()), *args],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip() or f"git {' '.join(args)} failed"
        raise ControlPlaneError(message)
    return completed.stdout.strip()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "candidate"


def _normalize_hypothesis(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _campaign_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_STATE_DIR / "campaign.json"


def _ideas_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_STATE_DIR / "ideas.json"


def _memory_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_STATE_DIR / "memory.json"


def _worker_manifest_path(repo_root: Path, worker_id: str) -> Path:
    return repo_root / DEFAULT_WORKERS_DIR / f"{worker_id}.json"


def _load_campaign(repo_root: Path) -> dict[str, Any]:
    return _read_json(_campaign_path(repo_root), {})


def _load_ideas(repo_root: Path) -> list[dict[str, Any]]:
    raw = _read_json(_ideas_path(repo_root), {"ideas": []})
    return list(raw.get("ideas", []))


def _save_ideas(repo_root: Path, ideas: list[dict[str, Any]]) -> None:
    _write_json(_ideas_path(repo_root), {"ideas": ideas})


def _load_memory(repo_root: Path) -> list[dict[str, Any]]:
    raw = _read_json(_memory_path(repo_root), {"notes": []})
    return list(raw.get("notes", []))


def _save_memory(repo_root: Path, notes: list[dict[str, Any]]) -> None:
    _write_json(_memory_path(repo_root), {"notes": notes})


def _load_worker_manifests(repo_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    worker_dir = repo_root / DEFAULT_WORKERS_DIR
    if not worker_dir.exists():
        return manifests
    for path in sorted(worker_dir.glob("*.json")):
        manifests.append(_read_json(path, {}))
    return manifests


def _save_worker_manifest(repo_root: Path, manifest: dict[str, Any]) -> None:
    _write_json(_worker_manifest_path(repo_root, manifest["worker_id"]), manifest)


def _load_worker_manifest(repo_root: Path, worker_id: str) -> dict[str, Any]:
    manifest = _read_json(_worker_manifest_path(repo_root, worker_id), None)
    if not manifest:
        raise ControlPlaneError(f"Unknown worker: {worker_id}")
    return manifest


def _load_results_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@contextmanager
def _results_lock(results_path: Path):
    init_results_tsv(results_path)
    lock_path = results_path.parent / f".{results_path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _parse_score(log_text: str) -> float:
    match = re.search(r"^\s*score\s+(-?\d+(?:\.\d+)?)", log_text, flags=re.MULTILINE)
    if match is None:
        raise ControlPlaneError("Unable to parse score from evaluator output")
    return float(match.group(1))


def _build_evaluation_command(repo_root: Path, runner_command: Sequence[str] | None) -> list[str]:
    if runner_command:
        return list(runner_command)
    return list(DEFAULT_EVALUATION_COMMAND)


def _changed_tracked_files(repo_root: Path) -> list[str]:
    output = _git(repo_root, "diff", "--name-only", "--relative", "HEAD")
    return sorted(path for path in output.splitlines() if path)


def _validate_mutable_change_set(repo_root: Path, mutable_file: Path) -> None:
    changed_files = _changed_tracked_files(repo_root)
    if not changed_files:
        raise ControlPlaneError("Worker has no tracked candidate change to evaluate")
    disallowed = [path for path in changed_files if path != str(mutable_file)]
    if disallowed:
        raise ControlPlaneError(
            "Worker changes must be limited to the mutable benchmark file; "
            f"found: {', '.join(disallowed)}"
        )


def init_control_plane(
    repo_root: str | Path = REPO_ROOT,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    campaign_path = _campaign_path(root)
    ideas_path = _ideas_path(root)
    memory_path = _memory_path(root)
    worker_dir = root / DEFAULT_WORKERS_DIR
    worktree_dir = root / DEFAULT_WORKTREE_DIR
    log_dir = root / DEFAULT_LOG_DIR
    notes_dir = root / "research/notes"
    for path in (campaign_path.parent, worker_dir, worktree_dir, log_dir, notes_dir):
        path.mkdir(parents=True, exist_ok=True)
    campaign = _read_json(campaign_path, {})
    if not campaign:
        campaign = {
            "created_at": _utcnow(),
            "max_workers": 2 if max_workers is None else max_workers,
            "benchmark": {
                "mutable_file": str(DEFAULT_MUTABLE_FILE),
                "results_path": str(DEFAULT_RESULTS_PATH),
                "evaluation_command": list(DEFAULT_EVALUATION_COMMAND),
                "docker_first": True,
            },
            "control_plane": {
                "worktree_dir": str(DEFAULT_WORKTREE_DIR),
                "log_dir": str(DEFAULT_LOG_DIR),
                "worker_branch_prefix": "worker",
            },
            "contracts": {
                "immutable_files": [
                    "autoresearch/experiments/restaurant_eval.py",
                    "autoresearch/tasks.py",
                    "program.md",
                    "AGENTS.md",
                ],
                "promotion_rule": "strict score improvement only",
                "evaluation_rule": "all runs go through Docker",
                "single_change_rule": "one worker evaluates one hypothesis",
            },
        }
    elif max_workers is not None:
        campaign["max_workers"] = max_workers
    _write_json(campaign_path, campaign)
    if not ideas_path.exists():
        _write_json(ideas_path, {"ideas": []})
    if not memory_path.exists():
        _write_json(memory_path, {"notes": []})
    return campaign


def add_experiment_idea(
    repo_root: str | Path,
    *,
    title: str,
    hypothesis: str,
    policy_family: str,
    rationale: str,
    author_role: str = "planner",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    init_control_plane(root)
    ideas = _load_ideas(root)
    slug = _slugify(title)
    fingerprint = hashlib.sha1(f"{title}|{hypothesis}|{policy_family}".encode("utf-8")).hexdigest()[:8]
    idea = {
        "idea_id": f"idea-{len(ideas) + 1:04d}-{slug[:20]}-{fingerprint}",
        "title": title,
        "hypothesis": hypothesis,
        "policy_family": policy_family,
        "rationale": rationale,
        "author_role": author_role,
        "status": "pending",
        "duplicate_of": None,
        "reviewer_note": "",
        "worker_id": None,
        "decision": None,
        "dedupe_key": _normalize_hypothesis(hypothesis or title),
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }
    ideas.append(idea)
    _save_ideas(root, ideas)
    return idea


def add_memory_note(
    repo_root: str | Path,
    *,
    title: str,
    body: str,
    tags: Sequence[str] | None = None,
    related_idea_id: str | None = None,
    related_worker_id: str | None = None,
    block_hypothesis: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    init_control_plane(root)
    notes = _load_memory(root)
    note = {
        "note_id": f"note-{len(notes) + 1:04d}-{hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]}",
        "title": title,
        "body": body,
        "tags": list(tags or []),
        "related_idea_id": related_idea_id,
        "related_worker_id": related_worker_id,
        "block_hypothesis": _normalize_hypothesis(block_hypothesis) if block_hypothesis else None,
        "created_at": _utcnow(),
    }
    notes.append(note)
    _save_memory(root, notes)
    return note


def review_experiment_ideas(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    init_control_plane(root)
    ideas = _load_ideas(root)
    notes = _load_memory(root)
    blocked = {note["block_hypothesis"]: note for note in notes if note.get("block_hypothesis")}
    approved_by_key: dict[str, str] = {}
    reviewed: list[dict[str, Any]] = []
    for idea in ideas:
        key = idea.get("dedupe_key") or _normalize_hypothesis(idea.get("hypothesis", ""))
        status = idea.get("status", "pending")
        if status in {"duplicate", "rejected"}:
            reviewed.append(idea)
            continue
        duplicate_of = approved_by_key.get(key)
        reviewer_note = ""
        if key in blocked:
            duplicate_of = blocked[key].get("related_idea_id") or blocked[key]["note_id"]
            status = "rejected"
            reviewer_note = f"Rejected by memory note {duplicate_of}"
        elif duplicate_of is not None:
            status = "duplicate"
            reviewer_note = f"Duplicate of {duplicate_of}"
        elif status == "pending":
            status = "approved"
            reviewer_note = "Approved for worker launch"
            approved_by_key[key] = idea["idea_id"]
        else:
            approved_by_key[key] = idea["idea_id"]
        idea = {
            **idea,
            "status": status,
            "duplicate_of": None if status == "approved" else duplicate_of,
            "reviewer_note": reviewer_note,
            "updated_at": _utcnow(),
        }
        if status in {"approved", "launched", "running", "completed", "kept", "promoted", "cleaned"}:
            approved_by_key[key] = idea["idea_id"]
        reviewed.append(idea)
    _save_ideas(root, reviewed)
    return reviewed


def launch_experiment_worker(
    repo_root: str | Path,
    *,
    idea_id: str,
    base_ref: str = "HEAD",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    campaign = init_control_plane(root)
    ideas = review_experiment_ideas(root)
    active_workers = [
        manifest for manifest in _load_worker_manifests(root) if manifest.get("status") not in FINAL_WORKER_STATUSES
    ]
    if len(active_workers) >= int(campaign.get("max_workers", 2)):
        raise ControlPlaneError("Worker cap reached; wait for an active worker to finish")
    idea = next((item for item in ideas if item["idea_id"] == idea_id), None)
    if idea is None:
        raise ControlPlaneError(f"Unknown idea: {idea_id}")
    if idea.get("status") != "approved":
        raise ControlPlaneError(f"Idea {idea_id} is not launchable (status={idea.get('status')})")
    worker_slug = _slugify(idea["title"])[:24]
    worker_hash = hashlib.sha1(f"{idea_id}|{_utcnow()}".encode("utf-8")).hexdigest()[:8]
    worker_id = f"worker-{worker_slug}-{worker_hash}"
    branch = f"{campaign['control_plane']['worker_branch_prefix']}/{worker_id}"
    worktree_path = root / DEFAULT_WORKTREE_DIR / worker_id
    if worktree_path.exists():
        raise ControlPlaneError(f"Worktree path already exists: {worktree_path}")
    _git(root, "worktree", "add", "-b", branch, str(worktree_path), base_ref)
    manifest = {
        "worker_id": worker_id,
        "idea_id": idea_id,
        "title": idea["title"],
        "hypothesis": idea["hypothesis"],
        "policy_family": idea["policy_family"],
        "branch": branch,
        "base_ref": base_ref,
        "worktree_path": str(worktree_path),
        "log_path": str((root / DEFAULT_LOG_DIR / f"{worker_id}.log")),
        "results_path": str((root / DEFAULT_RESULTS_PATH)),
        "mutable_file": str(DEFAULT_MUTABLE_FILE),
        "status": "launched",
        "decision": None,
        "score": None,
        "candidate_sha": None,
        "created_at": _utcnow(),
        "started_at": None,
        "finished_at": None,
    }
    _save_worker_manifest(root, manifest)
    for index, candidate in enumerate(ideas):
        if candidate["idea_id"] == idea_id:
            ideas[index] = {
                **candidate,
                "status": "launched",
                "worker_id": worker_id,
                "updated_at": _utcnow(),
            }
            break
    _save_ideas(root, ideas)
    return manifest


def _record_worker_result(
    *,
    results_path: Path,
    branch: str,
    sha: str,
    score: float | None,
    default_decision: str,
    keep_message: str,
    non_keep_message: str,
) -> tuple[str, float | None, float | None]:
    with _results_lock(results_path):
        current_best = read_best_result(results_path)
        best_score = None if current_best is None else float(current_best["score"])
        decision = default_decision
        message = non_keep_message.format(
            best_score="" if best_score is None else best_score
        )
        if score is not None and (best_score is None or score > best_score):
            decision = "keep"
            message = keep_message
        append_result(
            results_path,
            branch=branch,
            sha=sha,
            score=score,
            decision=decision,
            message=message,
        )
    return decision, score, best_score


def run_experiment_worker(
    repo_root: str | Path,
    *,
    worker_id: str,
    message: str,
    runner_command: Sequence[str] | None = None,
    cleanup: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    init_control_plane(root)
    manifest = _load_worker_manifest(root, worker_id)
    worktree_path = Path(manifest["worktree_path"])
    mutable_file = Path(manifest.get("mutable_file", DEFAULT_MUTABLE_FILE))
    results_path = Path(manifest.get("results_path", root / DEFAULT_RESULTS_PATH))
    log_path = Path(manifest["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_mutable_change_set(worktree_path, mutable_file)
    candidate_sha = commit_before_run(message, repo_root=worktree_path)
    command = _build_evaluation_command(root, runner_command)
    manifest.update({
        "status": "running",
        "candidate_sha": candidate_sha,
        "started_at": _utcnow(),
    })
    _save_worker_manifest(root, manifest)

    def _evaluate_once() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=worktree_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    completed = _evaluate_once()
    combined_output = completed.stdout
    if completed.returncode != 0:
        retry = _evaluate_once()
        combined_output = f"{combined_output}\n\n--- RETRY ---\n{retry.stdout}".strip()
        completed = retry
    log_path.write_text(combined_output + ("\n" if not combined_output.endswith("\n") else ""), encoding="utf-8")

    if completed.returncode != 0:
        revert_last_commit(repo_root=worktree_path)
        decision, _, _ = _record_worker_result(
            results_path=results_path,
            branch=manifest["branch"],
            sha=candidate_sha,
            score=None,
            default_decision="crash",
            keep_message="",
            non_keep_message=f"evaluation failed twice; log {_relative_to_repo(log_path, root)}",
        )
        manifest.update(
            {
                "status": "completed",
                "decision": decision,
                "score": None,
                "finished_at": _utcnow(),
            }
        )
    else:
        score = _parse_score(combined_output)
        decision, _, best_score = _record_worker_result(
            results_path=results_path,
            branch=manifest["branch"],
            sha=candidate_sha,
            score=score,
            default_decision="discard",
            keep_message=f"kept with score {score}; log {_relative_to_repo(log_path, root)}",
            non_keep_message=(
                f"discarded with score {score}; current best "
                f"{{best_score}}; log {_relative_to_repo(log_path, root)}"
            ),
        )
        if decision != "keep":
            revert_last_commit(repo_root=worktree_path)
        manifest.update(
            {
                "status": "completed",
                "decision": decision,
                "score": score,
                "finished_at": _utcnow(),
            }
        )
    _save_worker_manifest(root, manifest)
    ideas = _load_ideas(root)
    for index, idea in enumerate(ideas):
        if idea.get("worker_id") == worker_id:
            ideas[index] = {
                **idea,
                "status": "kept" if manifest["decision"] == "keep" else "completed",
                "decision": manifest["decision"],
                "updated_at": _utcnow(),
            }
            break
    _save_ideas(root, ideas)
    if cleanup and manifest["decision"] in {"discard", "crash"}:
        cleanup_experiment_worker(root, worker_id=worker_id)
        manifest = _load_worker_manifest(root, worker_id)
    return manifest


def promote_experiment_worker(repo_root: str | Path, *, worker_id: str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = _load_worker_manifest(root, worker_id)
    candidate_sha = manifest.get("candidate_sha")
    if manifest.get("decision") != "keep" or not candidate_sha:
        raise ControlPlaneError(f"Worker {worker_id} does not have a keep candidate to promote")
    _git(root, "cherry-pick", candidate_sha)
    manifest.update({"status": "promoted", "promoted_at": _utcnow()})
    _save_worker_manifest(root, manifest)
    ideas = _load_ideas(root)
    for index, idea in enumerate(ideas):
        if idea.get("worker_id") == worker_id:
            ideas[index] = {**idea, "status": "promoted", "updated_at": _utcnow()}
            break
    _save_ideas(root, ideas)
    return manifest


def cleanup_experiment_worker(
    repo_root: str | Path,
    *,
    worker_id: str,
    delete_branch: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = _load_worker_manifest(root, worker_id)
    worktree_path = Path(manifest["worktree_path"])
    if worktree_path.exists():
        _git(root, "worktree", "remove", "--force", str(worktree_path))
    if delete_branch:
        branches = _git(root, "branch", "--list", manifest["branch"])
        if branches:
            _git(root, "branch", "-D", manifest["branch"])
    manifest.update({"status": "cleaned", "cleaned_at": _utcnow()})
    _save_worker_manifest(root, manifest)
    ideas = _load_ideas(root)
    for index, idea in enumerate(ideas):
        if idea.get("worker_id") == worker_id:
            ideas[index] = {**idea, "status": "cleaned", "updated_at": _utcnow()}
            break
    _save_ideas(root, ideas)
    return manifest


def summarize_control_plane(repo_root: str | Path, *, recent_results: int = 5) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    init_control_plane(root)
    campaign = _load_campaign(root)
    ideas = _load_ideas(root)
    notes = _load_memory(root)
    workers = _load_worker_manifests(root)
    results_path = root / DEFAULT_RESULTS_PATH
    rows = _load_results_rows(results_path)
    active_workers = [worker for worker in workers if worker.get("status") not in FINAL_WORKER_STATUSES]
    completed_workers = [worker for worker in workers if worker.get("status") in FINAL_WORKER_STATUSES or worker.get("finished_at")]
    idea_counts: dict[str, int] = {}
    for idea in ideas:
        idea_counts[idea["status"]] = idea_counts.get(idea["status"], 0) + 1
    return {
        "campaign": campaign,
        "best_result": read_best_result(results_path),
        "recent_results": rows[-recent_results:],
        "active_workers": active_workers,
        "completed_workers": completed_workers,
        "idea_counts": idea_counts,
        "memory_note_count": len(notes),
    }


def _print_payload(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the local multi-agent control plane.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize control-plane state.")
    init_parser.add_argument("--max-workers", type=int, default=2)

    add_idea_parser = subparsers.add_parser("add-idea", help="Record a planner idea.")
    add_idea_parser.add_argument("--title", required=True)
    add_idea_parser.add_argument("--hypothesis", required=True)
    add_idea_parser.add_argument("--policy-family", required=True)
    add_idea_parser.add_argument("--rationale", required=True)
    add_idea_parser.add_argument("--author-role", default="planner")

    review_parser = subparsers.add_parser("review", help="Review pending ideas for duplication.")
    _ = review_parser

    add_memory_parser = subparsers.add_parser("add-memory", help="Record a durable memory note.")
    add_memory_parser.add_argument("--title", required=True)
    add_memory_parser.add_argument("--body", required=True)
    add_memory_parser.add_argument("--tag", action="append", dest="tags")
    add_memory_parser.add_argument("--related-idea-id")
    add_memory_parser.add_argument("--related-worker-id")
    add_memory_parser.add_argument("--block-hypothesis")

    launch_parser = subparsers.add_parser("launch-worker", help="Create an isolated worker worktree.")
    launch_parser.add_argument("--idea-id", required=True)
    launch_parser.add_argument("--base-ref", default="HEAD")

    run_parser = subparsers.add_parser("run-worker", help="Commit and evaluate a worker candidate.")
    run_parser.add_argument("--worker-id", required=True)
    run_parser.add_argument("--message", required=True)
    run_parser.add_argument("--cleanup", action="store_true")
    run_parser.add_argument(
        "--runner-command",
        help="Optional override command string used instead of the default Docker evaluator.",
    )

    promote_parser = subparsers.add_parser("promote-worker", help="Cherry-pick a kept worker commit.")
    promote_parser.add_argument("--worker-id", required=True)

    cleanup_parser = subparsers.add_parser("cleanup-worker", help="Remove a worker worktree.")
    cleanup_parser.add_argument("--worker-id", required=True)
    cleanup_parser.add_argument("--keep-branch", action="store_true")

    status_parser = subparsers.add_parser("status", help="Summarize active workers and frontier state.")
    status_parser.add_argument("--recent-results", type=int, default=5)

    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(args.repo_root).resolve()

    if args.command == "init":
        _print_payload(init_control_plane(repo_root, max_workers=args.max_workers))
        return 0
    if args.command == "add-idea":
        payload = add_experiment_idea(
            repo_root,
            title=args.title,
            hypothesis=args.hypothesis,
            policy_family=args.policy_family,
            rationale=args.rationale,
            author_role=args.author_role,
        )
        _print_payload(payload)
        return 0
    if args.command == "review":
        _print_payload({"ideas": review_experiment_ideas(repo_root)})
        return 0
    if args.command == "add-memory":
        payload = add_memory_note(
            repo_root,
            title=args.title,
            body=args.body,
            tags=args.tags,
            related_idea_id=args.related_idea_id,
            related_worker_id=args.related_worker_id,
            block_hypothesis=args.block_hypothesis,
        )
        _print_payload(payload)
        return 0
    if args.command == "launch-worker":
        _print_payload(launch_experiment_worker(repo_root, idea_id=args.idea_id, base_ref=args.base_ref))
        return 0
    if args.command == "run-worker":
        runner_command = shlex.split(args.runner_command) if args.runner_command else None
        _print_payload(
            run_experiment_worker(
                repo_root,
                worker_id=args.worker_id,
                message=args.message,
                runner_command=runner_command,
                cleanup=args.cleanup,
            )
        )
        return 0
    if args.command == "promote-worker":
        _print_payload(promote_experiment_worker(repo_root, worker_id=args.worker_id))
        return 0
    if args.command == "cleanup-worker":
        _print_payload(
            cleanup_experiment_worker(repo_root, worker_id=args.worker_id, delete_branch=not args.keep_branch)
        )
        return 0
    if args.command == "status":
        _print_payload(summarize_control_plane(repo_root, recent_results=args.recent_results))
        return 0
    raise ControlPlaneError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
