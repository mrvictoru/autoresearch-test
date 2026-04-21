from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


def _load_experiment_module(path: str | Path):
    target = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("mutation_experiment", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load experiment module from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_experiment(experiment_path: str | Path, *, seed: int = 13, samples: int = 128):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for neural mutation evaluation")
    assert torch is not None and nn is not None
    module = _load_experiment_module(experiment_path)
    train_fn = getattr(module, "train_and_predict", None)
    if train_fn is None:
        raise RuntimeError(
            f"Experiment module at {Path(experiment_path).resolve()} must define train_and_predict(...)"
        )
    default_config = getattr(module, "DEFAULT_CONFIG", {})
    torch.manual_seed(seed)
    x = torch.randn(samples, 2)
    y = ((x[:, 0] + x[:, 1]) > 0).long()
    split = int(samples * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    logits = train_fn(x_train=x_train, y_train=y_train, x_val=x_val, config=default_config)
    criterion = nn.CrossEntropyLoss()
    val_loss = criterion(logits, y_val).item()
    preds = logits.argmax(dim=1)
    accuracy = (preds == y_val).float().mean().item()
    score = accuracy - (0.05 * val_loss)
    return {
        "score": score,
        "validation_accuracy": accuracy,
        "validation_loss": val_loss,
    }


def _format_results_block(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "score",
        "validation_accuracy",
        "validation_loss",
        "training_seconds",
        "peak_vram_mb",
    ]
    lines = ["--- RESULTS ---"]
    seen = set()
    for key in ordered_keys:
        if key in metrics:
            lines.append(f"{key:18} {metrics[key]:.6f}")
            seen.add(key)
    for key in sorted(k for k in metrics.keys() if k not in seen):
        value = metrics[key]
        if isinstance(value, (int, float)):
            lines.append(f"{key:18} {float(value):.6f}")
        else:
            lines.append(f"{key:18} {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mutable neural experiment code.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="autoresearch/experiments/neural_train.py",
        help="Path to mutable experiment python file.",
    )
    args = parser.parse_args()
    metrics = evaluate_experiment(args.experiment)
    print(_format_results_block(metrics))
    print(f"METRIC_JSON: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
