from __future__ import annotations

from abc import ABC
from typing import Any

from .tasks import ResearchTask

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    HAS_TORCH = False


class NeuralTask(ResearchTask, ABC):
    """Base class for neural-model experimentation tasks."""

    def require_torch(self) -> None:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for neural tasks. Install with: pip install torch"
            )


class TinyTorchClassificationTask(NeuralTask):
    """Toy neural task for experimenting with architecture hyperparameters."""

    @property
    def name(self) -> str:
        return "tiny_torch_classification"

    def __init__(self, *, seed: int = 13, samples: int = 128) -> None:
        self.seed = seed
        self.samples = samples

    def describe_context(self) -> dict[str, Any]:
        return {
            "goal": "maximize validation accuracy on toy 2D classification",
            "model_family": "mlp",
            "supports": ["learning_rate", "num_layers", "num_heads"],
        }

    def initial_model_state(self) -> dict[str, Any]:
        return {
            "learning_rate": 1e-3,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "checkpoint_path": "in-memory",
        }

    def evaluate_model(self, model_state: dict[str, Any]) -> float:
        return float(self.evaluate_model_detailed(model_state)["score"])

    def evaluate_model_detailed(self, model_state: dict[str, Any]) -> dict[str, Any]:
        self.require_torch()
        assert torch is not None and nn is not None and optim is not None
        torch.manual_seed(self.seed)

        layers = max(1, int(model_state.get("num_layers", 2)))
        hidden_dim = max(4, int(model_state.get("hidden_dim", 16)))
        learning_rate = float(model_state.get("learning_rate", 1e-3))

        x = torch.randn(self.samples, 2)
        y = ((x[:, 0] + x[:, 1]) > 0).long()
        split = int(self.samples * 0.8)
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]

        modules: list[nn.Module] = []
        in_dim = 2
        for _ in range(layers):
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(nn.ReLU())
            in_dim = hidden_dim
        modules.append(nn.Linear(in_dim, 2))
        model = nn.Sequential(*modules)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).item()
            preds = val_logits.argmax(dim=1)
            accuracy = (preds == y_val).float().mean().item()

        score = accuracy - (0.05 * val_loss)
        return {
            "score": score,
            "validation_accuracy": accuracy,
            "validation_loss": val_loss,
        }
