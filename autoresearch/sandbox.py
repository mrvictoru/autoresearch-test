from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .mutation_agent import MutationProposal


@dataclass
class Workspace:
    source_root: Path
    session_root: Path
    frontier_root: Path
    _snapshot_counter: int = 0

    @classmethod
    def create(cls, *, source_root: str | Path) -> "Workspace":
        source = Path(source_root).resolve()
        session_root = Path(tempfile.mkdtemp(prefix="autoresearch-mutation-"))
        frontier_root = session_root / "frontier"
        shutil.copytree(source, frontier_root, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        return cls(source_root=source, session_root=session_root, frontier_root=frontier_root)

    def create_candidate(self) -> Path:
        candidate = self.session_root / "candidate"
        if candidate.exists():
            shutil.rmtree(candidate)
        shutil.copytree(self.frontier_root, candidate, ignore=shutil.ignore_patterns("__pycache__"))
        return candidate

    def promote_candidate(self, candidate_root: Path) -> None:
        if self.frontier_root.exists():
            shutil.rmtree(self.frontier_root)
        shutil.copytree(candidate_root, self.frontier_root, ignore=shutil.ignore_patterns("__pycache__"))

    def discard_candidate(self, candidate_root: Path) -> None:
        if candidate_root.exists():
            shutil.rmtree(candidate_root)

    def next_snapshot_id(self, *, root: Path, tracked_files: list[str]) -> str:
        self._snapshot_counter += 1
        digest = hashlib.sha256()
        for relative_path in sorted(tracked_files):
            target = (root / relative_path).resolve()
            digest.update(relative_path.encode("utf-8"))
            if target.exists():
                digest.update(target.read_bytes())
        return f"{self._snapshot_counter:04d}-{digest.hexdigest()[:12]}"

    def apply_proposal(
        self,
        *,
        candidate_root: Path,
        proposal: MutationProposal,
        allowed_mutable_files: list[str],
    ) -> None:
        allowed = set(allowed_mutable_files)
        touched = set(proposal.target_files)
        if proposal.edits:
            touched.update(edit.path for edit in proposal.edits)
        disallowed = sorted(path for path in touched if path not in allowed)
        if disallowed:
            raise PermissionError(f"Mutation attempted disallowed files: {', '.join(disallowed)}")
        if proposal.patch:
            patch_file = candidate_root / ".mutation.patch"
            patch_file.write_text(proposal.patch, encoding="utf-8")
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                cwd=candidate_root,
                check=True,
                capture_output=True,
                text=True,
            )
            patch_file.unlink(missing_ok=True)
        if proposal.edits:
            for edit in proposal.edits:
                target = candidate_root / edit.path
                target.parent.mkdir(parents=True, exist_ok=True)
                if edit.operation == "append" and target.exists():
                    target.write_text(
                        target.read_text(encoding="utf-8") + edit.content, encoding="utf-8"
                    )
                else:
                    target.write_text(edit.content, encoding="utf-8")
