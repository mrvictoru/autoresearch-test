from __future__ import annotations

from typing import Any

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


DEFAULT_CONFIG: dict[str, Any] = {
    "learning_rate": 1e-3,
    "num_layers": 2,
    "hidden_dim": 16,
    "epochs": 20,
    "activation": "relu",
    "optimizer": "adam",
}


def train_and_predict(
    *,
    x_train,
    y_train,
    x_val,
    config: dict[str, Any] | None = None,
):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for neural mutation experiments")
    assert torch is not None and nn is not None and optim is not None
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    layers = max(1, int(cfg.get("num_layers", 2)))
    hidden_dim = max(4, int(cfg.get("hidden_dim", 16)))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    epochs = max(1, int(cfg.get("epochs", 20)))
    activation_name = str(cfg.get("activation", "relu")).lower()
    activation = nn.Tanh if activation_name == "tanh" else nn.ReLU

    modules: list[nn.Module] = []
    in_dim = 2
    for _ in range(layers):
        modules.append(nn.Linear(in_dim, hidden_dim))
        modules.append(activation())
        in_dim = hidden_dim
    modules.append(nn.Linear(in_dim, 2))
    model = nn.Sequential(*modules)
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
    return val_logits
