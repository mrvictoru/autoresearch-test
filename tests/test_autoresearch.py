import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from autoresearch.brief import load_research_brief
from autoresearch.experiments.restaurant_eval import (
    _format_results_block as _format_restaurant_results_block,
)
from autoresearch.experiments.restaurant_eval import evaluate_experiment as evaluate_restaurant_experiment
from autoresearch.frontier import (
    append_result,
    commit_before_run,
    create_research_branch,
    get_current_sha,
    init_results_tsv,
    read_best_result,
    revert_last_commit,
)
from autoresearch.tasks import RestaurantInventoryTask


class AutoresearchFrameworkTests(unittest.TestCase):
    def _init_git_repo(self, root: Path) -> None:
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    def _commit_file(self, root: Path, relative_path: str, content: str, message: str) -> None:
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        subprocess.run(
            ["git", "add", relative_path],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_restaurant_task_emits_expected_metrics(self):
        metrics = RestaurantInventoryTask(days=30, seed=42).evaluate_model_detailed(
            {"reorder_point": 18, "target_stock": 40}
        )
        self.assertEqual(set(metrics), {"score", "stockouts", "waste_units", "total_orders"})
        self.assertEqual(metrics["score"], -965.0000000000001)
        self.assertEqual(metrics["stockouts"], 23.0)
        self.assertEqual(metrics["waste_units"], 335.0)
        self.assertEqual(metrics["total_orders"], 26.0)

    def test_restaurant_eval_results_block_format(self):
        rendered = _format_restaurant_results_block(
            {
                "score": -10.0,
                "stockouts": 2.0,
                "waste_units": 30.0,
                "total_orders": 4.0,
            }
        )
        self.assertIn("--- RESULTS ---", rendered)
        self.assertIn("score", rendered)
        self.assertIn("stockouts", rendered)
        self.assertIn("waste_units", rendered)
        self.assertIn("total_orders", rendered)

    def test_restaurant_eval_experiment_metrics(self):
        metrics = evaluate_restaurant_experiment("autoresearch/experiments/restaurant_train.py")
        self.assertEqual(
            metrics,
            {
                "score": -965.0000000000001,
                "stockouts": 23.0,
                "waste_units": 335.0,
                "total_orders": 26.0,
            },
        )

    def test_research_brief_loader_restaurant_json(self):
        brief = load_research_brief("research_brief_restaurant.json")
        self.assertEqual(brief.allowed_mutable_files, ["autoresearch/experiments/restaurant_train.py"])
        self.assertIn("autoresearch/tasks.py", brief.immutable_files)

    def test_research_brief_loader_restaurant_yaml(self):
        brief = load_research_brief("research_brief_restaurant.yaml")
        self.assertEqual(brief.constraints["experiment_command"][2], "autoresearch.experiments.restaurant_eval")

    def test_create_research_branch_creates_autoresearch_prefixed_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo(root)
            self._commit_file(root, "README.md", "seed\n", "init")

            branch_name = create_research_branch("Phase 3 / Restaurant", repo_root=root)
            self.assertEqual(branch_name, "autoresearch/phase-3-restaurant")
            head_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(head_branch, branch_name)

    def test_create_research_branch_rejects_empty_normalized_tag(self):
        with self.assertRaises(ValueError):
            create_research_branch("////")

    def test_commit_before_run_commits_tracked_changes_and_returns_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo(root)
            self._commit_file(root, "mutable.txt", "before\n", "init")

            (root / "mutable.txt").write_text("after\n", encoding="utf-8")
            (root / "results.tsv").write_text("scratch\n", encoding="utf-8")

            committed_sha = commit_before_run("candidate", repo_root=root)

            self.assertEqual(committed_sha, get_current_sha(repo_root=root))
            self.assertEqual((root / "mutable.txt").read_text(encoding="utf-8"), "after\n")
            untracked_files = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "results.tsv"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            self.assertEqual(untracked_files, ["results.tsv"])

    def test_revert_last_commit_restores_previous_frontier_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_repo(root)
            self._commit_file(root, "mutable.txt", "before\n", "init")
            previous_sha = get_current_sha(repo_root=root)

            (root / "mutable.txt").write_text("candidate\n", encoding="utf-8")
            commit_before_run("candidate", repo_root=root)
            reverted_sha = revert_last_commit(repo_root=root)

            self.assertEqual(reverted_sha, previous_sha)
            self.assertEqual((root / "mutable.txt").read_text(encoding="utf-8"), "before\n")

    def test_results_tsv_helpers_init_append_and_read_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.tsv"

            init_results_tsv(results_path)
            append_result(
                results_path,
                branch="autoresearch/run-a",
                sha="1111111",
                score=1.0,
                decision="keep",
                message="baseline",
            )
            append_result(
                results_path,
                branch="autoresearch/run-b",
                sha="2222222",
                score=2.0,
                decision="discard",
                message="worse frontier",
            )
            append_result(
                results_path,
                branch="autoresearch/run-c",
                sha="3333333",
                score=1.5,
                decision="keep",
                message="best frontier",
            )
            append_result(
                results_path,
                branch="autoresearch/run-d",
                sha="4444444",
                score=None,
                decision="crash",
                message="failed",
            )

            lines = results_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "timestamp\tbranch\tsha\tscore\tdecision\tmessage")
            best = read_best_result(results_path)
            self.assertIsNotNone(best)
            self.assertEqual(best["branch"], "autoresearch/run-c")
            self.assertEqual(best["sha"], "3333333")
            self.assertAlmostEqual(best["score"], 1.5)


if __name__ == "__main__":
    unittest.main()
