from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any
from urllib import request


class ResearchAgent(ABC):
    """Produces the next research suggestion for a task."""

    @abstractmethod
    def propose(
        self,
        *,
        task_name: str,
        model_state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """Return a plain-text proposal (ideally JSON-like parameters)."""


class LocalLLMResearchAgent(ResearchAgent):
    """Research agent backed by a local OpenAI-compatible API endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        system_prompt: str = "You are a research optimizer. Reply with concise parameter suggestions.",
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def propose(
        self,
        *,
        task_name: str,
        model_state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        user_prompt = (
            f"Task: {task_name}\n"
            f"Current model state: {json.dumps(model_state, sort_keys=True)}\n"
            f"Context: {json.dumps(context, sort_keys=True)}\n"
            "Suggest updated training parameters."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        return parsed["choices"][0]["message"]["content"].strip()
