import unittest
from pathlib import Path
import tempfile
import time
import json
import sys
import subprocess

from autoresearch.experiments.neural_eval import _format_results_block
from autoresearch.experiments.restaurant_eval import (
    _format_results_block as _format_restaurant_results_block,
)
from autoresearch.experiments.restaurant_eval import evaluate_experiment as evaluate_restaurant_experiment
from autoresearch.agent import (
    LocalLLMResearchAgent,
    PROMPT_TEMPLATE_PRESETS,
    ResearchAgent,
    TraceableAgent,
)
from autoresearch.brief import ResearchBrief, load_research_brief
from autoresearch.core import (
    AutoresearchRunner,
    ExperimentRecord,
    ExperimentStatus,
    MutationRunResult,
)
from autoresearch.executor import SafeExecutor
from autoresearch.frontier import create_research_branch
from autoresearch.harness import EvaluationHarness
from autoresearch.mutation_agent import FileEdit, MutationAgent, MutationProposal
from autoresearch.mutation_runner import MutationRunner
from autoresearch.sandbox import Workspace
from autoresearch.tasks import BlackjackTask, RestaurantInventoryTask
from autoresearch.training import (
    BlackjackPolicyTrainer,
    HyperparameterTrainer,
    InventoryPolicyTrainer,
    TrainerRegistry,
    _extract_int,
)


class SequenceAgent(ResearchAgent):
    def __init__(self, suggestions):
        self._suggestions = list(suggestions)
        self._index = 0

    def propose(self, *, task_name, model_state, context):
        suggestion = self._suggestions[self._index % len(self._suggestions)]
        self._index += 1
        return suggestion


class FakeMutationAgent(MutationAgent):
    def __init__(self, proposals):
        self._proposals = list(proposals)
        self._index = 0

    def propose_mutation(self, *, task_name, context):
        proposal = self._proposals[self._index % len(self._proposals)]
        self._index += 1
        return proposal


class FailingThenRecoveringMutationAgent(MutationAgent):
    def __init__(self, recovery_proposal: MutationProposal):
        self._first_call = True
        self._recovery_proposal = recovery_proposal

    def propose_mutation(self, *, task_name, context):
        if self._first_call:
            self._first_call = False
            raise ValueError("simulated agent failure")
        return self._recovery_proposal


class AutoresearchFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.registry = TrainerRegistry()
        self.registry.register("restaurant_inventory", InventoryPolicyTrainer())
        self.registry.register("blackjack", BlackjackPolicyTrainer())

    def test_inventory_runner_applies_suggestion(self):
        runner = AutoresearchRunner(
            agent=SequenceAgent(["reorder_point=24 target_stock=54"]),
            registry=self.registry,
        )
        result = runner.run(RestaurantInventoryTask(seed=10), iterations=1)
        self.assertEqual(result.history[0].model_state["reorder_point"], 24)
        self.assertEqual(result.history[0].model_state["target_stock"], 54)
        self.assertEqual(len(result.history), 1)

    def test_blackjack_trainer_clamps_threshold(self):
        runner = AutoresearchRunner(
            agent=SequenceAgent(["hit_threshold=99"]),
            registry=self.registry,
        )
        result = runner.run(BlackjackTask(seed=11), iterations=1)
        self.assertEqual(result.history[0].model_state["hit_threshold"], 20)

    def test_registry_requires_registered_task(self):
        with self.assertRaises(KeyError):
            TrainerRegistry().get("missing_task")

    def test_extract_int_uses_fallback_for_missing_or_malformed_values(self):
        self.assertEqual(_extract_int("no relevant key here", "hit_threshold", 16), 16)
        self.assertEqual(_extract_int("hit_threshold=abc", "hit_threshold", 16), 16)
        self.assertEqual(_extract_int("hit_threshold:18", "hit_threshold", 16), 18)
        self.assertEqual(_extract_int('{"hit_threshold": 19}', "hit_threshold", 16), 19)

    def test_runner_stores_detailed_metrics(self):
        runner = AutoresearchRunner(
            agent=SequenceAgent(["hit_threshold=17"]),
            registry=self.registry,
        )
        result = runner.run(BlackjackTask(seed=11), iterations=1)
        self.assertIn("wins", result.history[0].metrics)
        self.assertIn("score", result.history[0].metrics)

    def test_traceable_agent_captures_trace(self):
        runner = AutoresearchRunner(
            agent=TraceableAgent(SequenceAgent(["hit_threshold=17"])),
            registry=self.registry,
        )
        result = runner.run(BlackjackTask(seed=11), iterations=1)
        self.assertIsNotNone(result.history[0].latency_seconds)
        self.assertEqual(result.history[0].suggestion, "hit_threshold=17")

    def test_runresult_exports_csv_and_trace(self):
        runner = AutoresearchRunner(
            agent=SequenceAgent(["hit_threshold=17"]),
            registry=self.registry,
        )
        result = runner.run(BlackjackTask(seed=11), iterations=1)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "result.csv"
            trace_path = Path(tmp) / "trace.json"
            result.to_csv(csv_path)
            result.to_trace_log(trace_path)
            self.assertTrue(csv_path.exists())
            self.assertTrue(trace_path.exists())
            self.assertIn("iteration", csv_path.read_text(encoding="utf-8"))
            self.assertIn('"task_name"', trace_path.read_text(encoding="utf-8"))
            tsv_path = Path(tmp) / "result.tsv"
            result.to_csv(tsv_path, delimiter="\t")
            lines = tsv_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("\titeration\t", lines[0])
            self.assertIn("\t", lines[1])

    def test_prompt_preset_factory(self):
        agent = LocalLLMResearchAgent.from_preset(
            endpoint="http://localhost:8000",
            model="test-model",
            prompt_preset="concise",
        )
        self.assertEqual(agent.user_prompt_template, PROMPT_TEMPLATE_PRESETS["concise"])

    def test_hyperparameter_trainer_updates_values(self):
        updated = HyperparameterTrainer().train_step(
            model_state={"learning_rate": 0.001, "num_layers": 2, "num_heads": 2},
            suggestion='{"learning_rate": 0.01, "num_layers": 4, "num_heads": 8}',
            task_context={},
        )
        self.assertEqual(updated["num_layers"], 4)
        self.assertEqual(updated["num_heads"], 8)
        self.assertAlmostEqual(updated["learning_rate"], 0.01)

    def test_evaluation_harness_timeout(self):
        class SlowTask(RestaurantInventoryTask):
            def evaluate_model_detailed(self, model_state):  # type: ignore[override]
                time.sleep(0.05)
                return {"score": 1.0}

        harness = EvaluationHarness(timeout_seconds=0.01)
        with self.assertRaises(TimeoutError):
            harness.evaluate_model_detailed(task=SlowTask(), model_state={"reorder_point": 10})

    def test_research_brief_loader_json(self):
        payload = {
            "goal": "test",
            "constraints": {"timeout_seconds": 10},
            "allowed_mutable_files": ["a.py"],
            "immutable_files": ["b.py"],
            "time_budget_seconds": 120,
            "tie_breaker_policy": "best score",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brief.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            brief = load_research_brief(path)
        self.assertEqual(brief.goal, "test")
        self.assertEqual(brief.allowed_mutable_files, ["a.py"])

    def test_workspace_applies_whitelisted_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "repo"
            source.mkdir()
            target = source / "mutable.txt"
            target.write_text("hello", encoding="utf-8")
            workspace = Workspace.create(source_root=source)
            candidate = workspace.create_candidate()
            proposal = MutationProposal(
                description="replace",
                target_files=["mutable.txt"],
                edits=[FileEdit(path="mutable.txt", operation="replace", content="updated")],
            )
            workspace.apply_proposal(
                candidate_root=candidate,
                proposal=proposal,
                allowed_mutable_files=["mutable.txt"],
            )
            self.assertEqual((candidate / "mutable.txt").read_text(encoding="utf-8"), "updated")

    def test_workspace_rejects_disallowed_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "repo"
            source.mkdir()
            (source / "immutable.txt").write_text("keep", encoding="utf-8")
            workspace = Workspace.create(source_root=source)
            candidate = workspace.create_candidate()
            proposal = MutationProposal(
                description="bad",
                target_files=["immutable.txt"],
                edits=[FileEdit(path="immutable.txt", operation="replace", content="oops")],
            )
            with self.assertRaises(PermissionError):
                workspace.apply_proposal(
                    candidate_root=candidate,
                    proposal=proposal,
                    allowed_mutable_files=["mutable.txt"],
                )

    def test_safe_executor_extracts_metric_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.py"
            script.write_text('print(\'METRIC_JSON: {"score": 0.75}\')\n', encoding="utf-8")
            result = SafeExecutor(timeout_seconds=5).run(
                command=[sys.executable, str(script)],
                cwd=root,
                log_dir=root / "logs",
            )
            self.assertTrue(result.success)
            self.assertAlmostEqual(result.score or 0.0, 0.75)

    def test_mutation_run_result_exports(self):
        result = MutationRunResult(
            task_name="tiny_torch_classification",
            best_score=0.6,
            best_snapshot_id="snap",
            history=[
                ExperimentRecord(
                    iteration=1,
                    status=ExperimentStatus.KEEP,
                    description="improve",
                    snapshot_id="snap",
                    score=0.6,
                    run_log_path="run.log",
                    resource_metrics={"wall_time_seconds": 1.2},
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mutation.csv"
            log_path = Path(tmp) / "mutation.json"
            result.to_csv(csv_path)
            result.to_experiment_log(log_path)
            self.assertTrue(csv_path.exists())
            self.assertTrue(log_path.exists())
            self.assertIn("status", csv_path.read_text(encoding="utf-8"))
            self.assertIn('"best_snapshot_id"', log_path.read_text(encoding="utf-8"))

    def test_mutation_runner_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "score.txt").write_text("0.5", encoding="utf-8")
            (root / "eval.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        'score = float(Path("score.txt").read_text(encoding="utf-8").strip())',
                        'print(f\'METRIC_JSON: {{"score": {score}}}\')',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            brief = ResearchBrief(
                goal="improve score",
                constraints={"experiment_command": ["{python}", "eval.py"], "timeout_seconds": 5},
                allowed_mutable_files=["score.txt"],
                immutable_files=["eval.py"],
                time_budget_seconds=30,
                tie_breaker_policy="best score",
            )
            agent = FakeMutationAgent(
                [
                    MutationProposal(
                        description="increase score",
                        target_files=["score.txt"],
                        edits=[FileEdit(path="score.txt", operation="replace", content="0.6")],
                    ),
                    MutationProposal(
                        description="decrease score",
                        target_files=["score.txt"],
                        edits=[FileEdit(path="score.txt", operation="replace", content="0.4")],
                    ),
                ]
            )
            result = MutationRunner(
                agent=agent,
                executor=SafeExecutor(timeout_seconds=5),
            ).run(
                task_name="mutation_test",
                source_root=root,
                research_brief=brief,
                iterations=2,
            )
            self.assertEqual(len(result.history), 2)
            self.assertEqual(result.history[0].status, ExperimentStatus.KEEP)
            self.assertEqual(result.history[1].status, ExperimentStatus.DISCARD)
            self.assertAlmostEqual(result.best_score, 0.6)

    def test_mutation_runner_survives_agent_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "score.txt").write_text("0.5", encoding="utf-8")
            (root / "eval.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        'score = float(Path("score.txt").read_text(encoding="utf-8").strip())',
                        'print(f\'METRIC_JSON: {{"score": {score}}}\')',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            brief = ResearchBrief(
                goal="recover after agent failure",
                constraints={"experiment_command": ["{python}", "eval.py"], "timeout_seconds": 5},
                allowed_mutable_files=["score.txt"],
                immutable_files=["eval.py"],
                time_budget_seconds=30,
                tie_breaker_policy="best score",
            )
            result = MutationRunner(
                agent=FailingThenRecoveringMutationAgent(
                    MutationProposal(
                        description="recover with better score",
                        target_files=["score.txt"],
                        edits=[FileEdit(path="score.txt", operation="replace", content="0.8")],
                    )
                ),
                executor=SafeExecutor(timeout_seconds=5),
            ).run(
                task_name="mutation_recovery_test",
                source_root=root,
                research_brief=brief,
                iterations=2,
            )
            self.assertEqual(len(result.history), 2)
            self.assertEqual(result.history[0].status, ExperimentStatus.CRASH)
            self.assertEqual(result.history[0].failure_category, "ValueError")
            self.assertEqual(
                result.history[0].description,
                "<agent failed to produce proposal>",
            )
            self.assertEqual(result.history[1].status, ExperimentStatus.KEEP)
            self.assertAlmostEqual(result.best_score, 0.8)

    def test_mutation_runner_uses_wall_time_tie_breaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sleep.txt").write_text("0.20", encoding="utf-8")
            (root / "eval.py").write_text(
                "\n".join(
                    [
                        "import time",
                        "from pathlib import Path",
                        'sleep_seconds = float(Path("sleep.txt").read_text(encoding="utf-8").strip())',
                        "time.sleep(sleep_seconds)",
                        'print(\'METRIC_JSON: {"score": 0.5}\')',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            brief = ResearchBrief(
                goal="prefer faster runs when score ties",
                constraints={"experiment_command": ["{python}", "eval.py"], "timeout_seconds": 5},
                allowed_mutable_files=["sleep.txt"],
                immutable_files=["eval.py"],
                time_budget_seconds=30,
                # Canonical brief format expected by tie-breaker parsing:
                # "Prefer higher score; tie-breaker by lower wall_time_seconds."
                tie_breaker_policy="Prefer higher score; tie-break by lower wall_time_seconds.",
            )
            agent = FakeMutationAgent(
                [
                    MutationProposal(
                        description="same score faster",
                        target_files=["sleep.txt"],
                        edits=[FileEdit(path="sleep.txt", operation="replace", content="0.05")],
                    ),
                    MutationProposal(
                        description="same score slower",
                        target_files=["sleep.txt"],
                        edits=[FileEdit(path="sleep.txt", operation="replace", content="0.40")],
                    ),
                ]
            )
            result = MutationRunner(
                agent=agent,
                executor=SafeExecutor(timeout_seconds=5),
            ).run(
                task_name="mutation_tie_break_test",
                source_root=root,
                research_brief=brief,
                iterations=2,
            )
            self.assertEqual(len(result.history), 2)
            self.assertEqual(result.history[0].status, ExperimentStatus.KEEP)
            self.assertEqual(result.history[1].status, ExperimentStatus.DISCARD)
            self.assertAlmostEqual(result.best_score, 0.5)

    def test_cli_mutation_help(self):
        completed = subprocess.run(
            [sys.executable, "-m", "autoresearch", "mutation", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("mutation", completed.stdout.lower())

    def test_neural_eval_results_block_format(self):
        rendered = _format_results_block(
            {
                "score": 0.75,
                "validation_accuracy": 0.8,
                "validation_loss": 1.0,
            }
        )
        self.assertIn("--- RESULTS ---", rendered)
        self.assertIn("score", rendered)
        self.assertIn("validation_accuracy", rendered)
        self.assertIn("validation_loss", rendered)

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
        self.assertIn("score", metrics)
        self.assertIn("stockouts", metrics)
        self.assertIn("waste_units", metrics)
        self.assertIn("total_orders", metrics)

    def test_research_brief_loader_restaurant_json(self):
        brief = load_research_brief("research_brief_restaurant.json")
        self.assertIn("autoresearch/experiments/restaurant_train.py", brief.allowed_mutable_files)
        self.assertIn("autoresearch/experiments/restaurant_eval.py", brief.immutable_files)

    def test_create_research_branch_creates_autoresearch_prefixed_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            (root / "README.md").write_text("seed\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "README.md"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

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

    def test_create_research_branch_normalizes_mixed_case_and_special_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            (root / "README.md").write_text("seed\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "README.md"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            branch_name = create_research_branch("  PHASE /// 3__A  ", repo_root=root)
            self.assertEqual(branch_name, "autoresearch/phase-3__a")
            head_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(head_branch, branch_name)

    def test_create_research_branch_raises_when_branch_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            (root / "README.md").write_text("seed\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "README.md"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            base_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            create_research_branch("phase-3", repo_root=root)
            subprocess.run(
                ["git", "checkout", base_branch],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            with self.assertRaises(RuntimeError):
                create_research_branch("phase-3", repo_root=root)

    def test_create_research_branch_raises_when_not_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                create_research_branch("phase-3", repo_root=tmp)


if __name__ == "__main__":
    unittest.main()
