"""
Restaurant inventory policy experiment file.

This module defines candidate inventory policies.  The harness calls
``build_policy()`` to obtain the active policy for evaluation.

Included policies
-----------------
AdaptiveRestaurantPolicy
    Original rule-based heuristic (demand-forecast + safety-stock).

NeuralNetworkPolicy
    MLP-based policy trained via supervised learning on oracle labels
    derived from training-scenario frames.  Requires ``numpy`` and
    ``scikit-learn`` (pre-installed in the Docker image).

PolicyRegistry
    Lightweight registry so ``build_policy()`` can be changed to any
    named policy with a one-line edit.

Usage
-----
To switch policies, change the argument passed to ``PolicyRegistry.build``:

    return REGISTRY.build("neural_network")   # MLP policy
    return REGISTRY.build("adaptive")          # rule-based fallback
"""
from __future__ import annotations

import math
from typing import Sequence

from autoresearch.tasks import (
    IngredientSpec,
    RestaurantInventoryTask,
    RestaurantObservation,
    RestaurantPolicy,
    RestaurantScenario,
)

# ---------------------------------------------------------------------------
# Rule-based baseline policy
# ---------------------------------------------------------------------------

class AdaptiveRestaurantPolicy(RestaurantPolicy):
    """Demand-forecast heuristic with weekday-aware safety stock."""

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


# ---------------------------------------------------------------------------
# Neural network policy
# ---------------------------------------------------------------------------

def _extract_features(
    observation: RestaurantObservation,
    ingredient_names: list[str],
) -> list[float]:
    """Convert a ``RestaurantObservation`` into a flat numeric feature vector.

    Features
    --------
    * Cyclic day-of-week encoding (sin / cos)
    * Day-index progress (0 → 1 over the episode horizon)
    * Overall storage utilisation
    * Per-ingredient: inventory ratio, pipeline ratio, combined available
      ratio, recent-usage rate, lead time, shelf life, unit cost, waste cost
    """
    total_capacity = max(1, observation.total_storage_capacity)
    features: list[float] = [
        math.sin(2.0 * math.pi * observation.day_of_week / 7.0),
        math.cos(2.0 * math.pi * observation.day_of_week / 7.0),
        observation.day_index / 14.0,
        observation.current_storage_units / total_capacity,
    ]
    for name in ingredient_names:
        spec: IngredientSpec = observation.ingredient_specs[name]
        max_cap = max(1, spec.max_storage_units)
        inv = observation.inventory_on_hand.get(name, 0)
        pipe = observation.incoming_pipeline.get(name, 0)
        usage = observation.recent_ingredient_usage.get(name, 0.0)
        features.extend([
            inv / max_cap,
            pipe / max_cap,
            (inv + pipe) / max_cap,
            min(1.0, usage / (max_cap + 1e-6)),
            float(spec.lead_time_days) / 3.0,
            float(spec.shelf_life_days) / 10.0,
            float(spec.unit_cost) / 5.0,
            float(spec.waste_cost) / 5.0,
        ])
    return features


def _oracle_orders_for_day(
    *,
    day_index: int,
    scenario: RestaurantScenario,
    ingredient_specs: dict[str, IngredientSpec],
    task: RestaurantInventoryTask,
    inventory_on_hand: dict[str, int],
    incoming_pipeline: dict[str, int],
    total_storage_capacity: int,
) -> dict[str, int]:
    """Compute look-ahead orders using perfect knowledge of future demand.

    For each ingredient the desired stock is set to cover demand over the
    window ``[day_index, day_index + lead_time + 2)``.  The oracle therefore
    always orders ahead of incoming shortages while respecting per-ingredient
    and total storage hard caps.
    """
    frames_by_day: dict[int, list[dict[str, int]]] = {}
    for frame in scenario.frames:
        frames_by_day.setdefault(frame.day_index, []).append(frame.demand_by_item)

    # Accumulate future demand per ingredient for each spec's look-ahead window
    future_demand: dict[str, int] = {name: 0 for name in ingredient_specs}
    for spec in ingredient_specs.values():
        look_ahead = spec.lead_time_days + 2
        for future_day in range(day_index, min(scenario.days, day_index + look_ahead)):
            daily_frames = frames_by_day.get(future_day, [])
            combined: dict[str, int] = {}
            for demand_by_item in daily_frames:
                for item_name, qty in demand_by_item.items():
                    combined[item_name] = combined.get(item_name, 0) + qty
            usage = task.ingredient_usage_from_demand(combined)
            future_demand[spec.name] += usage.get(spec.name, 0)

    current_total = sum(inventory_on_hand.values()) + sum(incoming_pipeline.values())
    remaining_capacity = max(0, total_storage_capacity - current_total)

    orders: dict[str, int] = {}
    for spec in sorted(ingredient_specs.values(), key=lambda s: s.lead_time_days, reverse=True):
        available = inventory_on_hand.get(spec.name, 0) + incoming_pipeline.get(spec.name, 0)
        shortage = max(0, future_demand[spec.name] - available)
        if shortage <= 0 or remaining_capacity <= 0:
            continue
        headroom = spec.max_storage_units - available
        raw_order = min(shortage, headroom, remaining_capacity)
        if raw_order <= 0:
            continue
        order_multiple = max(1, spec.order_multiple)
        order = int(math.ceil(raw_order / order_multiple) * order_multiple)
        order = min(order, headroom, remaining_capacity)
        if order > 0:
            orders[spec.name] = order
            remaining_capacity -= order
    return orders


class NeuralNetworkPolicy(RestaurantPolicy):
    """Multi-layer perceptron inventory policy.

    Training procedure
    ------------------
    ``fit()`` simulates every training scenario **step by step**, computes
    oracle order quantities at each day (using perfect future-demand
    knowledge from the scenario frames), and collects
    ``(feature_vector, oracle_order_vector)`` pairs.  A single
    ``MLPRegressor`` is then trained to map observations to order
    quantities.  Capacity constraints are re-applied at inference time so
    the outputs always respect the benchmark's hard limits.

    Training simplification
    -----------------------
    The synthetic observations built during training set
    ``recent_ingredient_usage`` to an empty dict.  This avoids a
    distribution shift between the simplified training simulation
    (which does not model partial-fulfillment or lot-level ageing) and the
    real evaluator simulation.  The NN therefore learns to rely primarily
    on inventory-on-hand and pipeline state, which are tracked accurately
    during training.

    Parameters
    ----------
    hidden_layer_sizes:
        MLP hidden-layer architecture.  Default ``(128, 64, 32)`` provides
        enough capacity for the 12-ingredient feature space while training
        quickly on a few hundred samples.
    max_iter:
        Maximum SGD epochs passed to ``MLPRegressor``.
    random_state:
        Reproducibility seed.
    safety_margin:
        Multiplicative upward nudge applied to NN predictions before
        rounding, encouraging slight over-ordering to reduce stockouts.
    """

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
        max_iter: int = 500,
        random_state: int = 42,
        safety_margin: float = 0.0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.safety_margin = safety_margin
        self.ingredient_names: list[str] = []
        self._model = None  # sklearn MLPRegressor, set in fit()
        self._scaler = None  # sklearn StandardScaler, set in fit()
        self._trained: bool = False
        self._fallback = AdaptiveRestaurantPolicy()

    def fit(
        self,
        scenarios: Sequence[RestaurantScenario],
        task: RestaurantInventoryTask,
    ) -> None:
        """Train the MLP on oracle-labelled rollouts from training scenarios."""
        try:
            import numpy as np
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "NeuralNetworkPolicy requires numpy and scikit-learn.  "
                "Install them with: pip install numpy scikit-learn"
            ) from exc

        self._fallback.fit(scenarios, task)
        self.ingredient_names = sorted(spec.name for spec in task.ingredients)

        X_rows: list[list[float]] = []
        y_rows: list[list[float]] = []

        for scenario in scenarios:
            # Replay the scenario step-by-step to track inventory state
            # while generating oracle order labels from future-demand knowledge.
            # Start with the same initial inventory level as the real evaluator.
            # The floor of 8 units guards against near-zero initial stock for
            # small-capacity ingredients, matching _initial_lot_buckets behaviour.
            inventory_on_hand: dict[str, int] = {
                spec.name: max(8, spec.max_storage_units // 3)
                for spec in task.ingredients
            }
            incoming_pipeline: dict[str, dict[str, int]] = {}

            frames_by_day: dict[int, list[dict[str, int]]] = {}
            for frame in scenario.frames:
                frames_by_day.setdefault(frame.day_index, []).append(frame.demand_by_item)

            ingredient_specs = {spec.name: spec for spec in task.ingredients}

            for day_index in range(scenario.days):
                # Apply pipeline arrivals
                arriving = incoming_pipeline.pop(day_index, {})
                for name, qty in arriving.items():
                    inventory_on_hand[name] = inventory_on_hand.get(name, 0) + qty

                # Build pipeline totals (for observation)
                pipeline_totals: dict[str, int] = {}
                for deliveries in incoming_pipeline.values():
                    for name, qty in deliveries.items():
                        pipeline_totals[name] = pipeline_totals.get(name, 0) + qty

                # Compute oracle orders (uses future scenario frames)
                oracle_orders = _oracle_orders_for_day(
                    day_index=day_index,
                    scenario=scenario,
                    ingredient_specs=ingredient_specs,
                    task=task,
                    inventory_on_hand=dict(inventory_on_hand),
                    incoming_pipeline=dict(pipeline_totals),
                    total_storage_capacity=task.total_storage_capacity,
                )

                # Build a synthetic observation for feature extraction.
                # recent_ingredient_usage is omitted (empty) to avoid
                # distributional shift between the training simulation
                # (simplified) and the real evaluator simulation.
                current_storage = sum(inventory_on_hand.values())
                obs = RestaurantObservation(
                    day_index=day_index,
                    day_of_week=day_index % 7,
                    inventory_on_hand=dict(inventory_on_hand),
                    incoming_pipeline=dict(pipeline_totals),
                    recent_item_demand={},
                    recent_ingredient_usage={},
                    ingredient_specs=ingredient_specs,
                    menu_items={item.name: item for item in task.menu_items},
                    current_storage_units=current_storage,
                    total_storage_capacity=task.total_storage_capacity,
                )
                features = _extract_features(obs, self.ingredient_names)
                label = [float(oracle_orders.get(name, 0)) for name in self.ingredient_names]

                X_rows.append(features)
                y_rows.append(label)

                # Advance inventory: queue oracle orders then consume demand
                for name, qty in oracle_orders.items():
                    spec = ingredient_specs[name]
                    arrival_day = day_index + spec.lead_time_days
                    day_pipeline = incoming_pipeline.setdefault(arrival_day, {})
                    day_pipeline[name] = day_pipeline.get(name, 0) + qty

                # Consume demand (simplified: consume from on-hand)
                daily_frames = frames_by_day.get(day_index, [])
                combined_demand: dict[str, int] = {}
                for demand_by_item in daily_frames:
                    for item_name, qty in demand_by_item.items():
                        combined_demand[item_name] = combined_demand.get(item_name, 0) + qty
                usage = task.ingredient_usage_from_demand(combined_demand)
                for name, used in usage.items():
                    inventory_on_hand[name] = max(0, inventory_on_hand.get(name, 0) - used)

                # Simplified ageing: approximately 1/shelf_life_days fraction
                # expires per day.  Ingredients with shorter shelf lives lose
                # stock faster, matching the benchmark's FIFO lot logic.
                for spec in task.ingredients:
                    on_hand = inventory_on_hand.get(spec.name, 0)
                    daily_decay = on_hand // max(1, spec.shelf_life_days)
                    inventory_on_hand[spec.name] = max(0, on_hand - daily_decay)

        if not X_rows:
            return

        X = np.array(X_rows, dtype=float)
        y = np.array(y_rows, dtype=float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            tol=1e-4,
        )
        model.fit(X_scaled, y)

        self._scaler = scaler
        self._model = model
        self._trained = True

    def decide_orders(self, observation: RestaurantObservation) -> dict[str, int]:
        """Return capacity-constrained order quantities from MLP predictions."""
        if not self._trained or self._model is None or self._scaler is None:
            return self._fallback.decide_orders(observation)

        try:
            import numpy as np
        except ImportError:
            return self._fallback.decide_orders(observation)

        features = _extract_features(observation, self.ingredient_names)
        X = np.array([features], dtype=float)
        X_scaled = self._scaler.transform(X)
        raw: list[float] = self._model.predict(X_scaled)[0].tolist()

        # Apply safety margin then enforce capacity hard constraints
        current_total = (
            observation.current_storage_units
            + sum(observation.incoming_pipeline.values())
        )
        remaining_capacity = max(0, observation.total_storage_capacity - current_total)

        orders: dict[str, int] = {}
        # Process in descending lead-time order (same as evaluator's _normalize_orders)
        sorted_ingredients = sorted(
            self.ingredient_names,
            key=lambda n: observation.ingredient_specs[n].lead_time_days,
            reverse=True,
        )
        for name in sorted_ingredients:
            spec = observation.ingredient_specs[name]
            idx = self.ingredient_names.index(name)
            predicted = max(0.0, float(raw[idx])) * (1.0 + self.safety_margin)
            available = (
                observation.inventory_on_hand.get(name, 0)
                + observation.incoming_pipeline.get(name, 0)
            )
            headroom = spec.max_storage_units - available
            if headroom <= 0 or remaining_capacity <= 0:
                continue
            order_multiple = max(1, spec.order_multiple)
            suggested = int(math.ceil(predicted / order_multiple) * order_multiple)
            allowed = min(suggested, headroom, remaining_capacity)
            if allowed <= 0:
                continue
            orders[name] = allowed
            remaining_capacity -= allowed
        return orders


# ---------------------------------------------------------------------------
# Policy registry — change ``build_policy`` to switch active policy
# ---------------------------------------------------------------------------

class _PolicyRegistry:
    """Maps string names to factory callables for easy policy switching."""

    def __init__(self) -> None:
        self._factories: dict[str, type[RestaurantPolicy]] = {
            "adaptive": AdaptiveRestaurantPolicy,
            "neural_network": NeuralNetworkPolicy,
        }

    def register(self, name: str, policy_class: type[RestaurantPolicy]) -> None:
        """Register a custom policy class under *name*."""
        self._factories[name] = policy_class

    def build(self, name: str = "neural_network", **kwargs) -> RestaurantPolicy:
        """Instantiate the policy registered under *name*.

        Pass keyword arguments to the policy constructor via ``**kwargs``.
        """
        if name not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise ValueError(
                f"Unknown policy '{name}'. Available policies: {available}"
            )
        return self._factories[name](**kwargs)


REGISTRY = _PolicyRegistry()


def build_policy() -> RestaurantPolicy:
    """Return the active policy for the harness evaluator.

    Change the argument to ``REGISTRY.build(...)`` to experiment with
    different policy types:

        REGISTRY.build("neural_network")   — MLP policy (default)
        REGISTRY.build("adaptive")         — rule-based heuristic
        REGISTRY.build("neural_network", hidden_layer_sizes=(256, 128, 64))
    """
    return REGISTRY.build(
        "adaptive",
        safety_factor=1.8,
        freshness_bias=0.7,
        recent_demand_weight=0.55,
    )
