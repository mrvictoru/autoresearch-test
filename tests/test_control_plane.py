import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane import (
    add_experiment_idea,
    add_memory_note,
    cleanup_experiment_worker,
    init_control_plane,
    launch_experiment_worker,
    review_experiment_ideas,
    run_experiment_worker,
    summarize_control_plane,
)
from autoresearch.frontier import append_result, read_best_result


class ControlPlaneTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
        mutable_path = root / "autoresearch/experiments/restaurant_train.py"
        mutable_path.parent.mkdir(parents=True, exist_ok=True)
        mutable_path.write_text("print('baseline')\n", encoding="utf-8")
        (root / "README.md").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)

    def test_init_control_plane_creates_campaign_and_state_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)

            campaign = init_control_plane(root, max_workers=3)

            self.assertEqual(campaign["max_workers"], 3)
            self.assertTrue((root / "research/state/campaign.json").exists())
            self.assertTrue((root / "research/state/ideas.json").exists())
            self.assertTrue((root / "research/state/memory.json").exists())
            self.assertTrue((root / "artifacts/control-plane/worktrees").exists())

    def test_review_marks_duplicate_and_memory_blocked_ideas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            init_control_plane(root)
            first = add_experiment_idea(
                root,
                title="Lead time buffer",
                hypothesis="Increase the lead time safety buffer for long lead ingredients.",
                policy_family="heuristic reorder tuning",
                rationale="Improve service level on delayed deliveries.",
            )
            second = add_experiment_idea(
                root,
                title="Duplicate buffer",
                hypothesis="Increase the lead time safety buffer for long lead ingredients.",
                policy_family="heuristic reorder tuning",
                rationale="Same idea again.",
            )
            add_memory_note(
                root,
                title="Blocked ensemble retry",
                body="Skip this hypothesis for now.",
                block_hypothesis="Try the same ensemble blend again.",
            )
            blocked = add_experiment_idea(
                root,
                title="Blocked ensemble",
                hypothesis="Try the same ensemble blend again.",
                policy_family="ensemble",
                rationale="Should be rejected from memory.",
            )

            reviewed = {item["idea_id"]: item for item in review_experiment_ideas(root)}

            self.assertEqual(reviewed[first["idea_id"]]["status"], "approved")
            self.assertEqual(reviewed[second["idea_id"]]["status"], "duplicate")
            self.assertEqual(reviewed[second["idea_id"]]["duplicate_of"], first["idea_id"])
            self.assertEqual(reviewed[blocked["idea_id"]]["status"], "rejected")

    def test_launch_worker_creates_isolated_worktree_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            init_control_plane(root)
            idea = add_experiment_idea(
                root,
                title="Freshness cap",
                hypothesis="Reduce freshness bias for fast-spoiling items.",
                policy_family="inventory aging aware",
                rationale="Limit waste-heavy over-ordering.",
            )
            review_experiment_ideas(root)

            manifest = launch_experiment_worker(root, idea_id=idea["idea_id"])

            self.assertEqual(manifest["status"], "launched")
            self.assertTrue(Path(manifest["worktree_path"]).exists())
            self.assertTrue((root / f"research/state/workers/{manifest['worker_id']}.json").exists())

    def test_run_worker_records_frontier_and_reverts_discarded_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            init_control_plane(root)

            keep_idea = add_experiment_idea(
                root,
                title="Keep candidate",
                hypothesis="First worker should establish the frontier.",
                policy_family="heuristic reorder tuning",
                rationale="Seed a best score.",
            )
            review_experiment_ideas(root)
            keep_manifest = launch_experiment_worker(root, idea_id=keep_idea["idea_id"])
            keep_worktree = Path(keep_manifest["worktree_path"])
            mutable_path = keep_worktree / "autoresearch/experiments/restaurant_train.py"
            mutable_path.write_text("print('keep candidate')\n", encoding="utf-8")

            kept = run_experiment_worker(
                root,
                worker_id=keep_manifest["worker_id"],
                message="keep candidate",
                runner_command=[
                    sys.executable,
                    "-c",
                    "print('--- RESULTS ---'); print('score 5.0'); print('service_level 0.5'); print('METRIC_JSON: {}')",
                ],
            )

            self.assertEqual(kept["decision"], "keep")
            best = read_best_result(root / "results.tsv")
            self.assertIsNotNone(best)
            self.assertAlmostEqual(best["score"], 5.0)

            discard_idea = add_experiment_idea(
                root,
                title="Discard candidate",
                hypothesis="Second worker should lose against the best score.",
                policy_family="heuristic reorder tuning",
                rationale="Validate discard behavior.",
            )
            review_experiment_ideas(root)
            discard_manifest = launch_experiment_worker(root, idea_id=discard_idea["idea_id"])
            discard_worktree = Path(discard_manifest["worktree_path"])
            discard_mutable = discard_worktree / "autoresearch/experiments/restaurant_train.py"
            original = discard_mutable.read_text(encoding="utf-8")
            discard_mutable.write_text("print('discard candidate')\n", encoding="utf-8")

            discarded = run_experiment_worker(
                root,
                worker_id=discard_manifest["worker_id"],
                message="discard candidate",
                runner_command=[
                    sys.executable,
                    "-c",
                    "print('--- RESULTS ---'); print('score 4.0'); print('service_level 0.4'); print('METRIC_JSON: {}')",
                ],
            )

            self.assertEqual(discarded["decision"], "discard")
            self.assertEqual(discard_mutable.read_text(encoding="utf-8"), original)

    def test_status_summary_reports_best_result_and_active_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            init_control_plane(root)
            append_result(
                root / "results.tsv",
                branch="autoresearch/demo",
                sha="abc123",
                score=7.5,
                decision="keep",
                message="frontier",
            )
            idea = add_experiment_idea(
                root,
                title="Reporter candidate",
                hypothesis="Reporter should see launched workers.",
                policy_family="reporting",
                rationale="Exercise status view.",
            )
            review_experiment_ideas(root)
            manifest = launch_experiment_worker(root, idea_id=idea["idea_id"])

            summary = summarize_control_plane(root)

            self.assertAlmostEqual(summary["best_result"]["score"], 7.5)
            self.assertEqual(len(summary["active_workers"]), 1)
            self.assertEqual(summary["active_workers"][0]["worker_id"], manifest["worker_id"])

            cleaned = cleanup_experiment_worker(root, worker_id=manifest["worker_id"])
            self.assertEqual(cleaned["status"], "cleaned")


if __name__ == "__main__":
    unittest.main()
