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
    days: int = 30,
    seed: int = 42,
) -> dict[str, float]:
    module = _load_experiment_module(experiment_path)
    build_fn = getattr(module, "get_model_state", None)
    if build_fn is None:
        raise RuntimeError("Experiment module must define get_model_state()")
    model_state = build_fn()
    if not isinstance(model_state, dict):
        raise RuntimeError("get_model_state() must return a dict")
    task = RestaurantInventoryTask(days=days, seed=seed)
    metrics = task.evaluate_model_detailed(model_state)
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _format_results_block(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "score",
        "stockouts",
        "waste_units",
        "total_orders",
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
