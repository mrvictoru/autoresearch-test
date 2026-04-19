import unittest
from pathlib import Path
import tempfile

from autoresearch.agent import (
    LocalLLMResearchAgent,
    PROMPT_TEMPLATE_PRESETS,
    ResearchAgent,
    TraceableAgent,
)
from autoresearch.core import AutoresearchRunner
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


if __name__ == "__main__":
    unittest.main()
