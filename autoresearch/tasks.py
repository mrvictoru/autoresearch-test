from __future__ import annotations

import random
from abc import ABC, abstractmethod
from statistics import mean
from typing import Any


class ResearchTask(ABC):
    """Evaluates model state against a concrete optimization objective."""

    name: str

    @abstractmethod
    def describe_context(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def initial_model_state(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def evaluate_model(self, model_state: dict[str, Any]) -> float:
        """Return scalar score (higher is better)."""


class RestaurantInventoryTask(ResearchTask):
    name = "restaurant_inventory"

    def __init__(
        self,
        *,
        days: int = 14,
        demand_mean: int = 28,
        demand_std: int = 6,
        unit_cost: float = 1.0,
        waste_cost: float = 0.4,
        stockout_penalty: float = 2.0,
        seed: int = 7,
    ) -> None:
        self.days = days
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.unit_cost = unit_cost
        self.waste_cost = waste_cost
        self.stockout_penalty = stockout_penalty
        self.seed = seed

    def describe_context(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "demand_mean": self.demand_mean,
            "demand_std": self.demand_std,
            "goal": "minimize stockouts and waste",
        }

    def initial_model_state(self) -> dict[str, Any]:
        return {"reorder_point": 18, "target_stock": 40}

    def evaluate_model(self, model_state: dict[str, Any]) -> float:
        reorder_point = int(model_state["reorder_point"])
        target_stock = max(reorder_point + 1, int(model_state["target_stock"]))
        rng = random.Random(self.seed)
        stock = target_stock
        total_cost = 0.0
        for _ in range(self.days):
            demand = max(0, int(rng.gauss(self.demand_mean, self.demand_std)))
            sold = min(stock, demand)
            stockout = demand - sold
            stock -= sold
            total_cost += stockout * self.stockout_penalty
            total_cost += stock * self.waste_cost
            if stock <= reorder_point:
                order = target_stock - stock
                stock += order
                total_cost += order * self.unit_cost
        return -total_cost


class BlackjackTask(ResearchTask):
    name = "blackjack"

    def __init__(self, *, rounds: int = 300, seed: int = 17) -> None:
        self.rounds = rounds
        self.seed = seed

    def describe_context(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "actions": ["hit", "stand"],
            "goal": "maximize expected reward",
        }

    def initial_model_state(self) -> dict[str, Any]:
        return {"hit_threshold": 16}

    def evaluate_model(self, model_state: dict[str, Any]) -> float:
        threshold = int(model_state["hit_threshold"])
        rng = random.Random(self.seed)
        rewards = []
        for _ in range(self.rounds):
            player = self._draw(rng) + self._draw(rng)
            dealer = self._draw(rng) + self._draw(rng)
            while player < threshold:
                player += self._draw(rng)
                if player > 21:
                    rewards.append(-1.0)
                    break
            else:
                while dealer < 17:
                    dealer += self._draw(rng)
                if dealer > 21 or player > dealer:
                    rewards.append(1.0)
                elif player == dealer:
                    rewards.append(0.0)
                else:
                    rewards.append(-1.0)
        return mean(rewards)

    @staticmethod
    def _draw(rng: random.Random) -> int:
        return rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
