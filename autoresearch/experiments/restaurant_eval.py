from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from ..tasks import RestaurantInventoryTask


def _load_experiment_module(path: str | Path):
    target = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("mutation_experiment", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load experiment module from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_experiment(
    experiment_path: str | Path,
    *,
    days: int = 14,
    seed: int = 42,
) -> dict[str, float]:
    module = _load_experiment_module(experiment_path)
    build_fn = getattr(module, "build_policy", None)
    if build_fn is None:
        raise RuntimeError(
            f"Experiment module at {Path(experiment_path).resolve()} must define build_policy()"
        )
    policy = build_fn()
    if policy is None or not hasattr(policy, "decide_orders"):
        raise RuntimeError("build_policy() must return an object with decide_orders(...)")
    task = RestaurantInventoryTask(days=days, seed=seed)
    fit_fn = getattr(policy, "fit", None)
    if callable(fit_fn):
        fit_fn(task.training_scenarios(), task)
    metrics = task.evaluate_policy(policy, scenarios=task.validation_scenarios())
    return {key: float(value) for key, value in metrics.items()}


def _format_results_block(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "score",
        "service_level",
        "revenue",
        "fulfilled_orders",
        "lost_orders",
        "waste_units",
        "waste_cost",
        "holding_cost",
        "order_cost",
        "stockout_penalty",
    ]
    lines = ["--- RESULTS ---"]
    seen = set()
    for key in ordered_keys:
        if key in metrics:
            lines.append(f"{key:18} {metrics[key]:.6f}")
            seen.add(key)
    for key in sorted(k for k in metrics.keys() if k not in seen):
        lines.append(f"{key:18} {metrics[key]:.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mutable restaurant experiment code.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="autoresearch/experiments/restaurant_train.py",
        help="Path to mutable experiment python file.",
    )
    args = parser.parse_args()
    metrics = evaluate_experiment(args.experiment)
    print(_format_results_block(metrics))
    print(f"METRIC_JSON: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
