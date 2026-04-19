import unittest

from autoresearch.agent import ResearchAgent
from autoresearch.core import AutoresearchRunner
from autoresearch.tasks import BlackjackTask, RestaurantInventoryTask
from autoresearch.training import (
    BlackjackPolicyTrainer,
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


if __name__ == "__main__":
    unittest.main()
