from __future__ import annotations

import random
from typing import Any


class RestaurantInventoryTask:
    """Deterministic restaurant inventory benchmark."""

    @property
    def name(self) -> str:
        return "restaurant_inventory"

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
        return float(self.evaluate_model_detailed(model_state)["score"])

    def evaluate_model_detailed(self, model_state: dict[str, Any]) -> dict[str, float]:
        reorder_point = int(model_state["reorder_point"])
        target_stock = max(reorder_point + 1, int(model_state["target_stock"]))
        rng = random.Random(self.seed)
        stock = target_stock
        total_cost = 0.0
        total_stockouts = 0
        total_waste = 0
        total_orders = 0
        for _ in range(self.days):
            demand = max(0, int(rng.gauss(self.demand_mean, self.demand_std)))
            sold = min(stock, demand)
            stockout = demand - sold
            stock -= sold
            total_stockouts += stockout
            total_waste += stock
            total_cost += stockout * self.stockout_penalty
            total_cost += stock * self.waste_cost
            if stock <= reorder_point:
                order = target_stock - stock
                stock += order
                total_orders += 1
                total_cost += order * self.unit_cost
        return {
            "score": float(-total_cost),
            "stockouts": float(total_stockouts),
            "waste_units": float(total_waste),
            "total_orders": float(total_orders),
        }
