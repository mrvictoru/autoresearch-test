from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class IngredientSpec:
    name: str
    unit_cost: float
    waste_cost: float
    holding_cost: float
    shelf_life_days: int
    lead_time_days: int
    max_storage_units: int
    order_multiple: int = 1
    fixed_order_cost: float = 0.0


@dataclass(frozen=True)
class MenuItemSpec:
    name: str
    price: float
    recipe: dict[str, int]
    period_demand: dict[str, float]


@dataclass(frozen=True)
class DemandFrame:
    day_index: int
    period: str
    demand_by_item: dict[str, int]


@dataclass(frozen=True)
class RestaurantScenario:
    name: str
    seed: int
    days: int
    frames: tuple[DemandFrame, ...]


@dataclass(frozen=True)
class RestaurantObservation:
    day_index: int
    day_of_week: int
    inventory_on_hand: dict[str, int]
    incoming_pipeline: dict[str, int]
    recent_item_demand: dict[str, float]
    recent_ingredient_usage: dict[str, float]
    ingredient_specs: dict[str, IngredientSpec]
    menu_items: dict[str, MenuItemSpec]
    current_storage_units: int
    total_storage_capacity: int


class RestaurantPolicy(Protocol):
    def fit(
        self,
        scenarios: Sequence[RestaurantScenario],
        task: "RestaurantInventoryTask",
    ) -> None:
        ...

    def decide_orders(self, observation: RestaurantObservation) -> dict[str, int | float]:
        ...


class RestaurantInventoryTask:
    """Immutable multi-item restaurant inventory benchmark."""

    @property
    def name(self) -> str:
        return "restaurant_inventory"

    def __init__(
        self,
        *,
        days: int = 14,
        seed: int = 7,
        training_seeds: Sequence[int] | None = None,
        validation_seeds: Sequence[int] | None = None,
    ) -> None:
        self.days = days
        self.seed = seed
        self.periods = ("lunch", "dinner")
        self.total_storage_capacity = 1000
        self.stockout_penalty_per_order = 18.0
        self.ingredients = tuple(_default_ingredients())
        self.menu_items = tuple(_default_menu_items())
        self._ingredient_map = {spec.name: spec for spec in self.ingredients}
        self._menu_map = {item.name: item for item in self.menu_items}
        self.training_seeds = tuple(training_seeds or (seed + 11, seed + 23, seed + 37, seed + 53))
        self.validation_seeds = tuple(validation_seeds or (seed + 101, seed + 211, seed + 307))
        self._training_cache: tuple[RestaurantScenario, ...] | None = None
        self._validation_cache: tuple[RestaurantScenario, ...] | None = None

    def describe_context(self) -> dict[str, Any]:
        return {
            "goal": "maximize restaurant operating profit while avoiding stockouts, waste, and capacity violations",
            "days": self.days,
            "periods": list(self.periods),
            "ingredients": [spec.name for spec in self.ingredients],
            "menu_items": [item.name for item in self.menu_items],
            "train_scenarios": [scenario.name for scenario in self.training_scenarios()],
            "validation_scenarios": [scenario.name for scenario in self.validation_scenarios()],
            "constraints": {
                "overlapping_ingredients": True,
                "time_varying_demand": True,
                "ingredient_perishability": True,
                "supplier_lead_times": True,
                "storage_constraints": True,
            },
        }

    def training_scenarios(self) -> tuple[RestaurantScenario, ...]:
        if self._training_cache is None:
            self._training_cache = tuple(
                self._generate_scenario(name=f"train-{index + 1}", seed=seed)
                for index, seed in enumerate(self.training_seeds)
            )
        return self._training_cache

    def validation_scenarios(self) -> tuple[RestaurantScenario, ...]:
        if self._validation_cache is None:
            self._validation_cache = tuple(
                self._generate_scenario(name=f"val-{index + 1}", seed=seed)
                for index, seed in enumerate(self.validation_seeds)
            )
        return self._validation_cache

    def ingredient_usage_from_demand(self, demand_by_item: dict[str, int]) -> dict[str, int]:
        usage = {spec.name: 0 for spec in self.ingredients}
        for item_name, quantity in demand_by_item.items():
            recipe = self._menu_map[item_name].recipe
            for ingredient_name, units in recipe.items():
                usage[ingredient_name] += units * quantity
        return usage

    def evaluate_policy(
        self,
        policy: RestaurantPolicy,
        *,
        scenarios: Sequence[RestaurantScenario] | None = None,
    ) -> dict[str, float]:
        active_scenarios = tuple(scenarios or self.validation_scenarios())
        totals = {
            "score": 0.0,
            "revenue": 0.0,
            "service_level": 0.0,
            "fulfilled_orders": 0.0,
            "lost_orders": 0.0,
            "waste_units": 0.0,
            "waste_cost": 0.0,
            "holding_cost": 0.0,
            "order_cost": 0.0,
            "stockout_penalty": 0.0,
        }
        total_orders = 0.0
        for scenario in active_scenarios:
            metrics, _ = self._simulate_scenario(policy=policy, scenario=scenario)
            for key, value in metrics.items():
                totals[key] += value
            total_orders += metrics["fulfilled_orders"] + metrics["lost_orders"]
        totals["service_level"] = (
            totals["fulfilled_orders"] / total_orders if total_orders else 1.0
        )
        return totals

    def evaluate_policy_with_telemetry(
        self,
        policy: RestaurantPolicy,
        *,
        scenarios: Sequence[RestaurantScenario] | None = None,
        experiment_path: str | None = None,
        policy_name: str | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        active_scenarios = tuple(scenarios or self.validation_scenarios())
        totals = {
            "score": 0.0,
            "revenue": 0.0,
            "service_level": 0.0,
            "fulfilled_orders": 0.0,
            "lost_orders": 0.0,
            "waste_units": 0.0,
            "waste_cost": 0.0,
            "holding_cost": 0.0,
            "order_cost": 0.0,
            "stockout_penalty": 0.0,
        }
        total_orders = 0.0
        scenario_traces: list[dict[str, Any]] = []
        for scenario in active_scenarios:
            metrics, trace = self._simulate_scenario(policy=policy, scenario=scenario, capture_trace=True)
            for key, value in metrics.items():
                totals[key] += value
            total_orders += metrics["fulfilled_orders"] + metrics["lost_orders"]
            if trace is not None:
                scenario_traces.append(trace)
        totals["service_level"] = (
            totals["fulfilled_orders"] / total_orders if total_orders else 1.0
        )
        artifact = {
            "schema_version": "restaurant-trace-v1",
            "benchmark": self.describe_context(),
            "run": {
                "days": self.days,
                "seed": self.seed,
                "experiment_path": experiment_path,
                "policy_name": policy_name or type(policy).__name__,
            },
            "aggregate_metrics": dict(totals),
            "scenarios": scenario_traces,
        }
        return totals, artifact

    def _generate_scenario(self, *, name: str, seed: int) -> RestaurantScenario:
        rng = random.Random(seed)
        frames: list[DemandFrame] = []
        for day_index in range(self.days):
            day_of_week = day_index % 7
            weekend_multiplier = 1.18 if day_of_week in {4, 5} else 1.0
            early_week_multiplier = 0.92 if day_of_week == 0 else 1.0
            daily_multiplier = weekend_multiplier * early_week_multiplier
            if day_index >= max(1, self.days - 4):
                daily_multiplier *= 1.08
            for period in self.periods:
                period_multiplier = 1.0 if period == "lunch" else 1.22
                demand_by_item: dict[str, int] = {}
                for item in self.menu_items:
                    base_mean = item.period_demand[period] * daily_multiplier * period_multiplier
                    std_dev = max(1.0, base_mean * 0.35)
                    demand = max(0, int(round(rng.gauss(base_mean, std_dev))))
                    demand_by_item[item.name] = demand
                frames.append(
                    DemandFrame(
                        day_index=day_index,
                        period=period,
                        demand_by_item=demand_by_item,
                    )
                )
        return RestaurantScenario(name=name, seed=seed, days=self.days, frames=tuple(frames))

    def _simulate_scenario(
        self,
        *,
        policy: RestaurantPolicy,
        scenario: RestaurantScenario,
        capture_trace: bool = False,
    ) -> tuple[dict[str, float], dict[str, Any] | None]:
        inventory = {
            spec.name: _initial_lot_buckets(spec=spec, target_level=max(8, spec.max_storage_units // 3))
            for spec in self.ingredients
        }
        pipeline: dict[int, dict[str, int]] = {}
        recent_item_demands: list[dict[str, int]] = []
        recent_ingredient_usage: list[dict[str, int]] = []
        metrics = {
            "score": 0.0,
            "revenue": 0.0,
            "fulfilled_orders": 0.0,
            "lost_orders": 0.0,
            "waste_units": 0.0,
            "waste_cost": 0.0,
            "holding_cost": 0.0,
            "order_cost": 0.0,
            "stockout_penalty": 0.0,
        }
        trace: dict[str, Any] | None = None
        event_index = 0
        cumulative_cash = 0.0
        if capture_trace:
            trace = {
                "scenario": {
                    "name": scenario.name,
                    "seed": scenario.seed,
                    "days": scenario.days,
                },
                "events": [],
                "checkpoints": [],
            }

        def append_event(
            *,
            event_type: str,
            day_index: int,
            period: str | None,
            details: dict[str, Any],
            cash_delta: float = 0.0,
            inventory_before: dict[str, int] | None = None,
            inventory_after: dict[str, int] | None = None,
        ) -> None:
            nonlocal event_index, cumulative_cash
            if trace is None:
                return
            cumulative_cash += cash_delta
            trace["events"].append(
                {
                    "event_index": event_index,
                    "event_type": event_type,
                    "day_index": day_index,
                    "period": period,
                    "cash_delta": cash_delta,
                    "cumulative_cash": cumulative_cash,
                    "inventory_before": inventory_before,
                    "inventory_after": inventory_after,
                    "details": details,
                }
            )
            event_index += 1

        frames_by_day = _group_frames_by_day(scenario.frames)
        for day_index in range(scenario.days):
            arriving_today = pipeline.pop(day_index, {})
            inventory_before_arrival = _inventory_snapshot(inventory)
            for ingredient_name, quantity in arriving_today.items():
                inventory[ingredient_name][-1] += quantity
            if arriving_today:
                append_event(
                    event_type="restock_arrival",
                    day_index=day_index,
                    period=None,
                    details={"quantities": dict(arriving_today)},
                    inventory_before=inventory_before_arrival,
                    inventory_after=_inventory_snapshot(inventory),
                )

            observation = RestaurantObservation(
                day_index=day_index,
                day_of_week=day_index % 7,
                inventory_on_hand={name: _inventory_on_hand(lots) for name, lots in inventory.items()},
                incoming_pipeline=_pipeline_totals(pipeline),
                recent_item_demand=_average_recent_maps(recent_item_demands),
                recent_ingredient_usage=_average_recent_maps(recent_ingredient_usage),
                ingredient_specs=dict(self._ingredient_map),
                menu_items=dict(self._menu_map),
                current_storage_units=sum(_inventory_on_hand(lots) for lots in inventory.values()),
                total_storage_capacity=self.total_storage_capacity,
            )
            raw_orders = policy.decide_orders(observation) or {}
            normalized_orders = self._normalize_orders(
                requested_orders=raw_orders,
                inventory=inventory,
                pipeline=pipeline,
            )
            day_order_cost = 0.0
            order_details: dict[str, dict[str, int | float]] = {}
            for ingredient_name, quantity in normalized_orders.items():
                if quantity <= 0:
                    continue
                spec = self._ingredient_map[ingredient_name]
                arrival_day = day_index + spec.lead_time_days
                pipeline.setdefault(arrival_day, {})[ingredient_name] = (
                    pipeline.setdefault(arrival_day, {}).get(ingredient_name, 0) + quantity
                )
                ingredient_cost = quantity * spec.unit_cost + spec.fixed_order_cost
                metrics["order_cost"] += ingredient_cost
                day_order_cost += ingredient_cost
                order_details[ingredient_name] = {
                    "quantity": quantity,
                    "arrival_day": arrival_day,
                    "cost": ingredient_cost,
                }
            if order_details:
                append_event(
                    event_type="restock_order",
                    day_index=day_index,
                    period=None,
                    details={"orders": order_details},
                    cash_delta=-day_order_cost,
                    inventory_before=_inventory_snapshot(inventory),
                    inventory_after=_inventory_snapshot(inventory),
                )

            day_item_demand = {item.name: 0 for item in self.menu_items}
            day_usage = {ingredient.name: 0 for ingredient in self.ingredients}
            for frame in frames_by_day[day_index]:
                frame_fulfilled = 0.0
                frame_lost = 0.0
                for item_name, quantity in frame.demand_by_item.items():
                    day_item_demand[item_name] += quantity
                    item = self._menu_map[item_name]
                    for _ in range(quantity):
                        inventory_before_order = _inventory_snapshot(inventory)
                        if self._can_fulfill(inventory=inventory, recipe=item.recipe):
                            self._consume_recipe(inventory=inventory, recipe=item.recipe, usage=day_usage)
                            metrics["revenue"] += item.price
                            metrics["fulfilled_orders"] += 1.0
                            frame_fulfilled += 1.0
                            append_event(
                                event_type="customer_order",
                                day_index=day_index,
                                period=frame.period,
                                details={
                                    "item_name": item_name,
                                    "status": "fulfilled",
                                    "consumed_ingredients": dict(item.recipe),
                                    "missing_ingredients": {},
                                },
                                cash_delta=item.price,
                                inventory_before=inventory_before_order,
                                inventory_after=_inventory_snapshot(inventory),
                            )
                        else:
                            metrics["lost_orders"] += 1.0
                            frame_lost += 1.0
                            missing_ingredients = {
                                ingredient_name: units - inventory_before_order.get(ingredient_name, 0)
                                for ingredient_name, units in item.recipe.items()
                                if inventory_before_order.get(ingredient_name, 0) < units
                            }
                            append_event(
                                event_type="customer_order",
                                day_index=day_index,
                                period=frame.period,
                                details={
                                    "item_name": item_name,
                                    "status": "lost",
                                    "consumed_ingredients": {},
                                    "missing_ingredients": missing_ingredients,
                                },
                                cash_delta=-self.stockout_penalty_per_order,
                                inventory_before=inventory_before_order,
                                inventory_after=inventory_before_order,
                            )

                if trace is not None:
                    trace["checkpoints"].append(
                        {
                            "checkpoint_type": "period_end",
                            "day_index": day_index,
                            "period": frame.period,
                            "inventory": _inventory_snapshot(inventory),
                            "incoming_pipeline": _pipeline_totals(pipeline),
                            "cumulative_cash": cumulative_cash,
                            "demand_by_item": dict(frame.demand_by_item),
                            "fulfilled_orders": frame_fulfilled,
                            "lost_orders": frame_lost,
                        }
                    )

            recent_item_demands.append(day_item_demand)
            recent_ingredient_usage.append(day_usage)
            if len(recent_item_demands) > 5:
                recent_item_demands.pop(0)
            if len(recent_ingredient_usage) > 5:
                recent_ingredient_usage.pop(0)

            holding_cost = sum(
                _inventory_on_hand(inventory[spec.name]) * spec.holding_cost for spec in self.ingredients
            )
            metrics["holding_cost"] += holding_cost
            append_event(
                event_type="holding_cost",
                day_index=day_index,
                period=None,
                details={"holding_cost": holding_cost},
                cash_delta=-holding_cost,
                inventory_before=_inventory_snapshot(inventory),
                inventory_after=_inventory_snapshot(inventory),
            )
            expired_units, expired_cost, expired_by_ingredient = self._age_and_discard(inventory)
            metrics["waste_units"] += expired_units
            metrics["waste_cost"] += expired_cost
            if expired_units > 0:
                append_event(
                    event_type="spoilage",
                    day_index=day_index,
                    period=None,
                    details={"expired_by_ingredient": expired_by_ingredient},
                    cash_delta=-expired_cost,
                    inventory_before=_inventory_snapshot_with_aged_tail(inventory, expired_by_ingredient),
                    inventory_after=_inventory_snapshot(inventory),
                )
            if trace is not None:
                trace["checkpoints"].append(
                    {
                        "checkpoint_type": "day_close",
                        "day_index": day_index,
                        "period": None,
                        "inventory": _inventory_snapshot(inventory),
                        "incoming_pipeline": _pipeline_totals(pipeline),
                        "cumulative_cash": cumulative_cash,
                        "day_item_demand": dict(day_item_demand),
                        "day_ingredient_usage": dict(day_usage),
                        "metrics": dict(metrics),
                    }
                )

        metrics["stockout_penalty"] = metrics["lost_orders"] * self.stockout_penalty_per_order
        metrics["score"] = (
            metrics["revenue"]
            - metrics["order_cost"]
            - metrics["holding_cost"]
            - metrics["waste_cost"]
            - metrics["stockout_penalty"]
        )
        if trace is not None:
            trace["scenario_metrics"] = dict(metrics)
            trace["final_inventory"] = _inventory_snapshot(inventory)
        return metrics, trace

    def _normalize_orders(
        self,
        *,
        requested_orders: dict[str, int | float],
        inventory: dict[str, list[int]],
        pipeline: dict[int, dict[str, int]],
    ) -> dict[str, int]:
        on_hand = {name: _inventory_on_hand(lots) for name, lots in inventory.items()}
        incoming = _pipeline_totals(pipeline)
        normalized = {spec.name: 0 for spec in self.ingredients}
        current_total = sum(on_hand.values()) + sum(incoming.values())
        remaining_total_capacity = max(0, self.total_storage_capacity - current_total)
        for spec in sorted(
            self.ingredients,
            key=lambda s: (on_hand[s.name] + incoming.get(s.name, 0)) / max(1, s.max_storage_units),
        ):
            raw_value = requested_orders.get(spec.name, 0)
            requested = max(0, int(round(float(raw_value))))
            if requested <= 0:
                continue
            max_for_ingredient = max(0, spec.max_storage_units - on_hand[spec.name] - incoming.get(spec.name, 0))
            allowed = min(requested, max_for_ingredient, remaining_total_capacity)
            if allowed <= 0:
                continue
            if spec.order_multiple > 1:
                allowed = int(math.ceil(allowed / spec.order_multiple) * spec.order_multiple)
                allowed = min(allowed, max_for_ingredient, remaining_total_capacity)
            normalized[spec.name] = max(0, allowed)
            remaining_total_capacity = max(0, remaining_total_capacity - normalized[spec.name])
        return normalized

    def _can_fulfill(self, *, inventory: dict[str, list[int]], recipe: dict[str, int]) -> bool:
        return all(_inventory_on_hand(inventory[ingredient_name]) >= units for ingredient_name, units in recipe.items())

    def _consume_recipe(
        self,
        *,
        inventory: dict[str, list[int]],
        recipe: dict[str, int],
        usage: dict[str, int],
    ) -> None:
        for ingredient_name, units in recipe.items():
            _consume_units(inventory[ingredient_name], units)
            usage[ingredient_name] += units

    def _age_and_discard(self, inventory: dict[str, list[int]]) -> tuple[float, float, dict[str, int]]:
        total_units = 0.0
        total_cost = 0.0
        expired_by_ingredient: dict[str, int] = {}
        for spec in self.ingredients:
            lots = inventory[spec.name]
            expired = lots[0]
            total_units += float(expired)
            total_cost += float(expired) * spec.waste_cost
            if expired > 0:
                expired_by_ingredient[spec.name] = expired
            inventory[spec.name] = lots[1:] + [0]
        return total_units, total_cost, expired_by_ingredient


def _group_frames_by_day(frames: Sequence[DemandFrame]) -> dict[int, list[DemandFrame]]:
    grouped: dict[int, list[DemandFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.day_index, []).append(frame)
    return grouped


def _initial_lot_buckets(*, spec: IngredientSpec, target_level: int) -> list[int]:
    lots = [0 for _ in range(spec.shelf_life_days)]
    lots[-1] = min(spec.max_storage_units, target_level)
    return lots


def _inventory_on_hand(lots: Sequence[int]) -> int:
    return int(sum(lots))


def _inventory_snapshot(inventory: dict[str, list[int]]) -> dict[str, int]:
    return {name: _inventory_on_hand(lots) for name, lots in inventory.items()}


def _inventory_snapshot_with_aged_tail(
    inventory: dict[str, list[int]],
    expired_by_ingredient: dict[str, int],
) -> dict[str, int]:
    snapshot = _inventory_snapshot(inventory)
    for ingredient_name, quantity in expired_by_ingredient.items():
        snapshot[ingredient_name] = snapshot.get(ingredient_name, 0) + quantity
    return snapshot


def _consume_units(lots: list[int], quantity: int) -> None:
    remaining = quantity
    for index in range(len(lots)):
        if remaining <= 0:
            return
        take = min(lots[index], remaining)
        lots[index] -= take
        remaining -= take


def _pipeline_totals(pipeline: dict[int, dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for deliveries in pipeline.values():
        for ingredient_name, quantity in deliveries.items():
            totals[ingredient_name] = totals.get(ingredient_name, 0) + quantity
    return totals


def _average_recent_maps(rows: Sequence[dict[str, int]]) -> dict[str, float]:
    if not rows:
        return {}
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / len(rows) for key, value in totals.items()}


def _default_ingredients() -> list[IngredientSpec]:
    return [
        IngredientSpec("bun", unit_cost=0.8, waste_cost=0.3, holding_cost=0.05, shelf_life_days=4, lead_time_days=1, max_storage_units=80, fixed_order_cost=1.5),
        IngredientSpec("beef_patty", unit_cost=2.4, waste_cost=1.2, holding_cost=0.12, shelf_life_days=3, lead_time_days=2, max_storage_units=70, fixed_order_cost=2.5),
        IngredientSpec("chicken", unit_cost=1.9, waste_cost=0.9, holding_cost=0.09, shelf_life_days=3, lead_time_days=2, max_storage_units=65, fixed_order_cost=2.0),
        IngredientSpec("tortilla", unit_cost=0.6, waste_cost=0.2, holding_cost=0.04, shelf_life_days=5, lead_time_days=1, max_storage_units=120, fixed_order_cost=1.0),
        IngredientSpec("lettuce", unit_cost=0.5, waste_cost=0.35, holding_cost=0.05, shelf_life_days=2, lead_time_days=1, max_storage_units=130, fixed_order_cost=1.2),
        IngredientSpec("tomato", unit_cost=0.4, waste_cost=0.28, holding_cost=0.04, shelf_life_days=3, lead_time_days=1, max_storage_units=320, fixed_order_cost=1.1),
        IngredientSpec("cheese", unit_cost=0.9, waste_cost=0.35, holding_cost=0.05, shelf_life_days=5, lead_time_days=2, max_storage_units=210, fixed_order_cost=1.3),
        IngredientSpec("dough", unit_cost=1.0, waste_cost=0.45, holding_cost=0.05, shelf_life_days=3, lead_time_days=1, max_storage_units=45, fixed_order_cost=1.4),
        IngredientSpec("pasta", unit_cost=0.7, waste_cost=0.2, holding_cost=0.03, shelf_life_days=6, lead_time_days=2, max_storage_units=90, fixed_order_cost=1.1),
        IngredientSpec("broth", unit_cost=0.6, waste_cost=0.18, holding_cost=0.03, shelf_life_days=6, lead_time_days=2, max_storage_units=65, fixed_order_cost=1.0),
        IngredientSpec("onion", unit_cost=0.25, waste_cost=0.1, holding_cost=0.02, shelf_life_days=7, lead_time_days=2, max_storage_units=160, fixed_order_cost=0.8),
        IngredientSpec("croutons", unit_cost=0.3, waste_cost=0.08, holding_cost=0.02, shelf_life_days=10, lead_time_days=2, max_storage_units=40, fixed_order_cost=0.6),
    ]


def _default_menu_items() -> list[MenuItemSpec]:
    return [
        MenuItemSpec(
            name="burger",
            price=13.5,
            recipe={"bun": 1, "beef_patty": 1, "cheese": 1, "lettuce": 1, "tomato": 1, "onion": 1},
            period_demand={"lunch": 7.5, "dinner": 9.0},
        ),
        MenuItemSpec(
            name="chicken_wrap",
            price=11.5,
            recipe={"tortilla": 1, "chicken": 1, "lettuce": 1, "tomato": 1, "cheese": 1},
            period_demand={"lunch": 6.0, "dinner": 5.0},
        ),
        MenuItemSpec(
            name="pasta",
            price=14.0,
            recipe={"pasta": 2, "tomato": 2, "cheese": 1, "onion": 1},
            period_demand={"lunch": 4.5, "dinner": 7.0},
        ),
        MenuItemSpec(
            name="pizza",
            price=16.0,
            recipe={"dough": 1, "tomato": 3, "cheese": 2, "onion": 1},
            period_demand={"lunch": 3.5, "dinner": 8.5},
        ),
        MenuItemSpec(
            name="salad",
            price=10.5,
            recipe={"lettuce": 2, "tomato": 2, "onion": 1, "cheese": 1, "croutons": 1},
            period_demand={"lunch": 5.5, "dinner": 4.0},
        ),
        MenuItemSpec(
            name="soup",
            price=9.0,
            recipe={"broth": 2, "chicken": 1, "onion": 1, "tomato": 1},
            period_demand={"lunch": 3.0, "dinner": 2.5},
        ),
    ]
