from __future__ import annotations

import math
from typing import Sequence

from autoresearch.tasks import RestaurantInventoryTask, RestaurantObservation, RestaurantPolicy, RestaurantScenario


class AdaptiveRestaurantPolicy(RestaurantPolicy):
    def __init__(
        self,
        *,
        safety_factor: float = 1.35,
        freshness_bias: float = 0.95,
        recent_demand_weight: float = 0.35,
    ) -> None:
        self.safety_factor = safety_factor
        self.freshness_bias = freshness_bias
        self.recent_demand_weight = recent_demand_weight
        self.average_usage_by_weekday: dict[int, dict[str, float]] = {}
        self.global_average_usage: dict[str, float] = {}

    def fit(
        self,
        scenarios: Sequence[RestaurantScenario],
        task: RestaurantInventoryTask,
    ) -> None:
        weekday_totals = {day: {ingredient.name: 0.0 for ingredient in task.ingredients} for day in range(7)}
        weekday_counts = {day: 0 for day in range(7)}
        global_totals = {ingredient.name: 0.0 for ingredient in task.ingredients}
        global_days = 0
        for scenario in scenarios:
            frames_by_day: dict[int, list[dict[str, int]]] = {}
            for frame in scenario.frames:
                frames_by_day.setdefault(frame.day_index, []).append(frame.demand_by_item)
            for day_index, daily_demands in frames_by_day.items():
                combined_item_demand: dict[str, int] = {}
                for demand_by_item in daily_demands:
                    for item_name, quantity in demand_by_item.items():
                        combined_item_demand[item_name] = combined_item_demand.get(item_name, 0) + quantity
                ingredient_usage = task.ingredient_usage_from_demand(combined_item_demand)
                day_of_week = day_index % 7
                weekday_counts[day_of_week] += 1
                global_days += 1
                for ingredient_name, quantity in ingredient_usage.items():
                    weekday_totals[day_of_week][ingredient_name] += float(quantity)
                    global_totals[ingredient_name] += float(quantity)
        self.average_usage_by_weekday = {
            day: {
                ingredient: (
                    weekday_totals[day][ingredient] / weekday_counts[day] if weekday_counts[day] else 0.0
                )
                for ingredient in weekday_totals[day]
            }
            for day in weekday_totals
        }
        self.global_average_usage = {
            ingredient: (global_totals[ingredient] / global_days if global_days else 0.0)
            for ingredient in global_totals
        }

    def decide_orders(self, observation: RestaurantObservation) -> dict[str, int]:
        current_total = observation.current_storage_units + sum(observation.incoming_pipeline.values())
        remaining_total_capacity = max(0, observation.total_storage_capacity - current_total)
        recent_usage = observation.recent_ingredient_usage
        learned_usage = self.average_usage_by_weekday.get(
            observation.day_of_week,
            self.global_average_usage,
        )
        orders: dict[str, int] = {}
        for ingredient_name, spec in sorted(
            observation.ingredient_specs.items(),
            key=lambda item: item[1].lead_time_days,
            reverse=True,
        ):
            baseline_daily = learned_usage.get(ingredient_name, self.global_average_usage.get(ingredient_name, 0.0))
            observed_daily = recent_usage.get(ingredient_name, baseline_daily)
            blended_daily = (
                (1.0 - self.recent_demand_weight) * baseline_daily
                + self.recent_demand_weight * observed_daily
            )
            coverage_days = max(1.0, float(spec.lead_time_days + 1))
            target_stock = blended_daily * coverage_days + self.safety_factor * math.sqrt(max(1.0, blended_daily))
            freshness_cap = blended_daily * max(1.0, (spec.shelf_life_days - 0.25) * self.freshness_bias)
            desired_units = min(spec.max_storage_units, int(round(max(target_stock, freshness_cap))))
            available_units = observation.inventory_on_hand.get(ingredient_name, 0) + observation.incoming_pipeline.get(ingredient_name, 0)
            shortage = max(0, desired_units - available_units)
            if shortage <= 0 or remaining_total_capacity <= 0:
                continue
            order_multiple = max(1, spec.order_multiple)
            suggested = int(math.ceil(shortage / order_multiple) * order_multiple)
            allowed = min(suggested, spec.max_storage_units - available_units, remaining_total_capacity)
            if allowed <= 0:
                continue
            orders[ingredient_name] = allowed
            remaining_total_capacity = max(0, remaining_total_capacity - allowed)
        return orders


def build_policy() -> RestaurantPolicy:
    return AdaptiveRestaurantPolicy()
