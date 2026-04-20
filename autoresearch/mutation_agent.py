from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib import request

from .agent import AgentTrace, ResearchAgent

MUTATION_PROMPT_TEMPLATE = (
    "Task: {task_name}\n"
    "Context JSON: {context_json}\n"
    "Return JSON object with keys: description, target_files, and one of patch or edits.\n"
    "If using edits, each edit must include: path, operation, content."
)
MAX_ERROR_PAYLOAD_LENGTH = 200


@dataclass
class FileEdit:
    path: str
    operation: str
    content: str


@dataclass
class MutationProposal:
    description: str
    target_files: list[str]
    patch: str | None = None
    edits: list[FileEdit] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "description": self.description,
            "target_files": self.target_files,
        }
        if self.patch:
            payload["patch"] = self.patch
        if self.edits:
            payload["edits"] = [
                {"path": item.path, "operation": item.operation, "content": item.content}
                for item in self.edits
            ]
        return payload


class MutationAgent(ResearchAgent, ABC):
    @abstractmethod
    def propose_mutation(
        self,
        *,
        task_name: str,
        context: dict[str, Any],
    ) -> MutationProposal:
        pass

    def propose(
        self,
        *,
        task_name: str,
        model_state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        proposal = self.propose_mutation(task_name=task_name, context=context)
        return json.dumps(proposal.to_dict(), sort_keys=True)


class LocalLLMMutationAgent(MutationAgent):
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        system_prompt: str = "You are a mutation research agent. Return JSON only.",
        user_prompt_template: str = MUTATION_PROMPT_TEMPLATE,
        temperature: float = 0.2,
        timeout_seconds: int = 45,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._last_trace: AgentTrace | None = None

    def _build_prompt(self, *, task_name: str, context: dict[str, Any]) -> str:
        return self.user_prompt_template.format(
            task_name=task_name,
            context=context,
            context_json=json.dumps(context, sort_keys=True),
        )

    def propose_mutation(
        self,
        *,
        task_name: str,
        context: dict[str, Any],
    ) -> MutationProposal:
        user_prompt = self._build_prompt(task_name=task_name, context=context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
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
            content = parsed["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise ValueError(
                f"Unexpected response payload from local LLM endpoint: {exc}. "
                f"Payload snippet: {raw[:MAX_ERROR_PAYLOAD_LENGTH]}"
            ) from exc
        proposal_payload = _parse_json_from_text(content)
        proposal = _proposal_from_payload(proposal_payload)
        self._last_trace = AgentTrace(
            task_name=task_name,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            raw_response=raw,
            suggestion=json.dumps(proposal.to_dict(), sort_keys=True),
            latency_seconds=latency_seconds,
        )
        return proposal

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
        raise ValueError("Mutation agent expects a JSON object response")
    return parsed


def _proposal_from_payload(payload: dict[str, Any]) -> MutationProposal:
    description = str(payload.get("description", "")).strip()
    target_files = payload.get("target_files", [])
    if not isinstance(target_files, list) or not all(isinstance(p, str) for p in target_files):
        raise ValueError("Mutation proposal 'target_files' must be a list of strings")
    patch = payload.get("patch")
    edits_payload = payload.get("edits")
    edits: list[FileEdit] | None = None
    if edits_payload is not None:
        if not isinstance(edits_payload, list):
            raise ValueError("Mutation proposal 'edits' must be a list")
        edits = []
        for item in edits_payload:
            if not isinstance(item, dict):
                raise ValueError("Each edit must be a JSON object")
            edits.append(
                FileEdit(
                    path=str(item.get("path", "")),
                    operation=str(item.get("operation", "replace")),
                    content=str(item.get("content", "")),
                )
            )
    if not patch and not edits:
        raise ValueError("Mutation proposal must include either 'patch' or 'edits'")
    return MutationProposal(
        description=description or "mutation proposal",
        target_files=[str(path) for path in target_files],
        patch=str(patch) if patch is not None else None,
        edits=edits,
    )
