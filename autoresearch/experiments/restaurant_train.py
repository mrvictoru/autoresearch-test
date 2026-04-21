from __future__ import annotations

from typing import Any


DEFAULT_MODEL_STATE: dict[str, Any] = {
    "reorder_point": 18,
    "target_stock": 40,
}


def get_model_state() -> dict[str, Any]:
    return dict(DEFAULT_MODEL_STATE)
