from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any
from urllib import request

MAX_ERROR_PAYLOAD_LENGTH = 200
PROMPT_TEMPLATE_PRESETS: dict[str, str] = {
    "concise": (
        "Task: {task_name}\n"
        "Current model state: {model_state_json}\n"
        "Context: {context_json}\n"
        "Suggest updated training parameters."
    ),
    "chain-of-thought": (
        "Task: {task_name}\n"
        "Current model state: {model_state_json}\n"
        "Context: {context_json}\n"
        "Reason briefly about tradeoffs, then provide updated training parameters."
    ),
    "json-only": (
        "Task: {task_name}\n"
        "Current model state: {model_state_json}\n"
        "Context: {context_json}\n"
        "Return only valid JSON with updated training parameters."
    ),
}


@dataclass
class AgentTrace:
    task_name: str
    system_prompt: str | None
    user_prompt: str | None
    raw_response: str
    suggestion: str
    latency_seconds: float


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

    def get_last_trace(self) -> AgentTrace | None:
        return None


class LocalLLMResearchAgent(ResearchAgent):
    """Research agent backed by a local OpenAI-compatible API endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        system_prompt: str = "You are a research optimizer. Reply with concise parameter suggestions.",
        user_prompt_template: str = PROMPT_TEMPLATE_PRESETS["concise"],
        temperature: float = 0.2,
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._last_trace: AgentTrace | None = None

    @classmethod
    def from_preset(
        cls,
        *,
        endpoint: str,
        model: str,
        prompt_preset: str = "concise",
        system_prompt: str = "You are a research optimizer. Reply with concise parameter suggestions.",
        temperature: float = 0.2,
        timeout_seconds: int = 30,
    ) -> "LocalLLMResearchAgent":
        if prompt_preset not in PROMPT_TEMPLATE_PRESETS:
            raise ValueError(
                f"Unknown prompt preset '{prompt_preset}'. "
                f"Available presets: {', '.join(sorted(PROMPT_TEMPLATE_PRESETS))}"
            )
        return cls(
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            user_prompt_template=PROMPT_TEMPLATE_PRESETS[prompt_preset],
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    def _build_user_prompt(
        self, *, task_name: str, model_state: dict[str, Any], context: dict[str, Any]
    ) -> str:
        try:
            return self.user_prompt_template.format(
                task_name=task_name,
                model_state=model_state,
                context=context,
                model_state_json=json.dumps(model_state, sort_keys=True),
                context_json=json.dumps(context, sort_keys=True),
            )
        except KeyError as exc:
            raise ValueError(
                f"user_prompt_template contains unknown placeholder: {exc}"
            ) from exc

    def _build_messages(
        self, *, task_name: str, model_state: dict[str, Any], context: dict[str, Any]
    ) -> list[dict[str, str]]:
        user_prompt = self._build_user_prompt(
            task_name=task_name, model_state=model_state, context=context
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def propose(
        self,
        *,
        task_name: str,
        model_state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        messages = self._build_messages(
            task_name=task_name, model_state=model_state, context=context
        )
        user_prompt = messages[-1]["content"] if messages else None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        latency_seconds = time.perf_counter() - start
        parsed = json.loads(raw)
        try:
            suggestion = parsed["choices"][0]["message"]["content"].strip()
            self._last_trace = AgentTrace(
                task_name=task_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                raw_response=raw,
                suggestion=suggestion,
                latency_seconds=latency_seconds,
            )
            return suggestion
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise ValueError(
                f"Unexpected response payload from local LLM endpoint: {exc}. "
                f"Payload snippet: {raw[:MAX_ERROR_PAYLOAD_LENGTH]}"
            ) from exc

    def get_last_trace(self) -> AgentTrace | None:
        return self._last_trace


class FewShotResearchAgent(LocalLLMResearchAgent):
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        few_shot_examples: list[tuple[dict[str, Any], str]] | None = None,
        system_prompt: str = "You are a research optimizer. Reply with concise parameter suggestions.",
        user_prompt_template: str = PROMPT_TEMPLATE_PRESETS["concise"],
        temperature: float = 0.2,
        timeout_seconds: int = 30,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        self.few_shot_examples = few_shot_examples or []

    def _build_messages(
        self, *, task_name: str, model_state: dict[str, Any], context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for example_state, example_suggestion in self.few_shot_examples:
            messages.append(
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        task_name=task_name, model_state=example_state, context=context
                    ),
                }
            )
            messages.append({"role": "assistant", "content": example_suggestion})
        messages.append(
            {
                "role": "user",
                "content": self._build_user_prompt(
                    task_name=task_name, model_state=model_state, context=context
                ),
            }
        )
        return messages


class StructuredOutputAgent(LocalLLMResearchAgent):
    """Agent variant that normalizes responses into strict JSON."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        system_prompt: str = "You are a research optimizer. Reply with JSON only.",
        user_prompt_template: str = PROMPT_TEMPLATE_PRESETS["json-only"],
        temperature: float = 0.2,
        timeout_seconds: int = 30,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    def propose(
        self,
        *,
        task_name: str,
        model_state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        raw_suggestion = super().propose(
            task_name=task_name, model_state=model_state, context=context
        )
        parsed = _parse_json_from_text(raw_suggestion)
        suggestion = json.dumps(parsed, sort_keys=True)
        last_trace = self.get_last_trace()
        if last_trace is not None:
            self._last_trace = AgentTrace(
                task_name=last_trace.task_name,
                system_prompt=last_trace.system_prompt,
                user_prompt=last_trace.user_prompt,
                raw_response=last_trace.raw_response,
                suggestion=suggestion,
                latency_seconds=last_trace.latency_seconds,
            )
        return suggestion


class TraceableAgent(ResearchAgent):
    """Decorator that captures per-call traces for any wrapped ResearchAgent."""

    def __init__(self, wrapped: ResearchAgent) -> None:
        self.wrapped = wrapped
        self.traces: list[AgentTrace] = []
        self._last_trace: AgentTrace | None = None

    def propose(
        self,
        *,
        task_name: str,
        model_state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        start = time.perf_counter()
        suggestion = self.wrapped.propose(
            task_name=task_name, model_state=model_state, context=context
        )
        latency_seconds = time.perf_counter() - start
        wrapped_trace = self.wrapped.get_last_trace()
        trace = (
            wrapped_trace
            if wrapped_trace is not None
            else AgentTrace(
                task_name=task_name,
                system_prompt=None,
                user_prompt=None,
                raw_response=suggestion,
                suggestion=suggestion,
                latency_seconds=latency_seconds,
            )
        )
        self._last_trace = trace
        self.traces.append(trace)
        return suggestion

    def get_last_trace(self) -> AgentTrace | None:
        return self._last_trace


def _parse_json_from_text(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("StructuredOutputAgent expects a JSON object response")
    return parsed
