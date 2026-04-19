from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import (
    LocalLLMResearchAgent,
    PROMPT_TEMPLATE_PRESETS,
    ResearchAgent,
    TraceableAgent,
)
from .core import AutoresearchRunner
from .tasks import BlackjackTask, RestaurantInventoryTask
from .training import BlackjackPolicyTrainer, InventoryPolicyTrainer, TrainerRegistry
from .visualise import plot_run_result, save_html_report


class CyclicDemoAgent(ResearchAgent):
    """Deterministic stub for local demo without requiring an LLM server."""

    def __init__(self) -> None:
        self._counter = 0

    def propose(self, *, task_name: str, model_state: dict, context: dict) -> str:
        self._counter += 1
        if task_name == "restaurant_inventory":
            return "reorder_point=22 target_stock=52"
        return f"hit_threshold={16 + (self._counter % 3)}"


def _build_registry() -> TrainerRegistry:
    registry = TrainerRegistry()
    registry.register("restaurant_inventory", InventoryPolicyTrainer())
    registry.register("blackjack", BlackjackPolicyTrainer())
    return registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoresearch demo experiments.")
    parser.add_argument("--config", type=str, help="Path to JSON/YAML experiment config.")
    parser.add_argument(
        "--task",
        choices=["restaurant_inventory", "blackjack", "all"],
        default="all",
        help="Task to run.",
    )
    parser.add_argument("--iterations", type=int, default=4, help="Number of iterations.")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print full per-iteration trace (prompt, response, parsed state).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory for CSV, trace log, HTML report, and manifest outputs.",
    )
    parser.add_argument("--agent-endpoint", type=str, help="Local OpenAI-compatible endpoint.")
    parser.add_argument("--agent-model", type=str, help="Model name at local endpoint.")
    parser.add_argument(
        "--prompt-preset",
        choices=sorted(PROMPT_TEMPLATE_PRESETS.keys()),
        default="concise",
        help="Prompt template preset for local LLM agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for LocalLLMResearchAgent.",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(content)
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from exc
        return yaml.safe_load(content) or {}
    raise ValueError("Unsupported config format. Use .json, .yml, or .yaml")


def _deep_get(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config else default


def _build_agent(config: dict[str, Any], args: argparse.Namespace) -> ResearchAgent:
    endpoint = args.agent_endpoint or _deep_get(config.get("agent", {}), "endpoint", None)
    model = args.agent_model or _deep_get(config.get("agent", {}), "model", None)
    if endpoint and model:
        system_prompt = _deep_get(
            config.get("agent", {}),
            "system_prompt",
            "You are a research optimizer. Reply with concise parameter suggestions.",
        )
        prompt_preset = _deep_get(config.get("agent", {}), "prompt_preset", args.prompt_preset)
        temperature = float(_deep_get(config.get("agent", {}), "temperature", args.temperature))
        return LocalLLMResearchAgent.from_preset(
            endpoint=endpoint,
            model=model,
            prompt_preset=prompt_preset,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    return CyclicDemoAgent()


def _build_tasks(config: dict[str, Any], args: argparse.Namespace):
    configured_task = _deep_get(config, "task", args.task)
    task_params = config.get("task_params", {})
    if configured_task == "restaurant_inventory":
        return [RestaurantInventoryTask(**task_params)]
    if configured_task == "blackjack":
        return [BlackjackTask(**task_params)]
    return [RestaurantInventoryTask(), BlackjackTask()]


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return output or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _write_manifest(
    *,
    path: Path,
    config: dict[str, Any],
    task_name: str,
    iterations: int,
    best_score: float,
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task_name,
        "iterations": iterations,
        "best_score": best_score,
        "git_commit": _git_commit_hash(),
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_trace(task_name: str, result) -> None:
    print(f"\nTrace for task={task_name}:")
    for item in result.history:
        print(f"  Iteration {item.iteration}")
        print(f"    Suggestion: {item.suggestion}")
        print(f"    Score: {item.score:.6f}")
        print(f"    Metrics: {item.metrics}")
        if item.prompt_sent is not None:
            print(f"    Prompt: {item.prompt_sent}")
        if item.raw_response is not None:
            print(f"    Raw response: {item.raw_response}")
        if item.latency_seconds is not None:
            print(f"    Latency (s): {item.latency_seconds:.4f}")


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config) if args.config else {}
    iterations = int(_deep_get(config, "iterations", args.iterations))
    output_dir = Path(_deep_get(config, "output_dir", args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = _build_registry()
    agent = TraceableAgent(_build_agent(config, args))
    runner = AutoresearchRunner(agent=agent, registry=registry)

    for task in _build_tasks(config, args):
        result = runner.run(task, iterations=iterations)
        print(
            f"{task.name} best: {result.best_model_state} "
            f"(score: {result.best_score:.6f})"
        )
        task_dir = output_dir / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(task_dir / "results.csv")
        result.to_trace_log(task_dir / "trace.json")
        save_html_report(result, task_dir / "report.html")
        _write_manifest(
            path=task_dir / "manifest.json",
            config=config or {"task": task.name},
            task_name=task.name,
            iterations=iterations,
            best_score=result.best_score,
        )
        if args.trace:
            _print_trace(task.name, result)
        if args.plot:
            plot_run_result(result)


if __name__ == "__main__":
    main()
