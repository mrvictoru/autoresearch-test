from __future__ import annotations

from .agent import ResearchAgent
from .core import AutoresearchRunner
from .tasks import BlackjackTask, RestaurantInventoryTask
from .training import BlackjackPolicyTrainer, InventoryPolicyTrainer, TrainerRegistry


class CyclicDemoAgent(ResearchAgent):
    """Deterministic stub for local demo without requiring an LLM server."""

    def __init__(self) -> None:
        self._counter = 0

    def propose(self, *, task_name: str, model_state: dict, context: dict) -> str:
        self._counter += 1
        if task_name == "restaurant_inventory":
            return "reorder_point=22 target_stock=52"
        return f"hit_threshold={16 + (self._counter % 3)}"


def main() -> None:
    registry = TrainerRegistry()
    registry.register("restaurant_inventory", InventoryPolicyTrainer())
    registry.register("blackjack", BlackjackPolicyTrainer())
    runner = AutoresearchRunner(agent=CyclicDemoAgent(), registry=registry)

    inventory_result = runner.run(RestaurantInventoryTask(), iterations=4)
    blackjack_result = runner.run(BlackjackTask(), iterations=4)
    print("Inventory best:", inventory_result.best_model_state, inventory_result.best_score)
    print("Blackjack best:", blackjack_result.best_model_state, blackjack_result.best_score)


if __name__ == "__main__":
    main()
