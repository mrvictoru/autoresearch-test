from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .core import MutationRunResult, RunResult


def plot_run_result(
    result: RunResult,
    *,
    include_parameters: bool = True,
    show: bool = True,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    iterations = [entry.iteration for entry in result.history]
    scores = [entry.score for entry in result.history]
    parameter_keys = sorted(
        {key for entry in result.history for key in entry.model_state.keys()}
    )

    if include_parameters and parameter_keys:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        score_ax = axes[0]
        param_ax = axes[1]
    else:
        fig, score_ax = plt.subplots(1, 1, figsize=(10, 4))
        param_ax = None

    score_ax.plot(iterations, scores, marker="o", label="score")
    score_ax.set_title(f"{result.task_name}: score by iteration")
    score_ax.set_ylabel("score")
    score_ax.grid(True, alpha=0.3)

    if param_ax is not None:
        for key in parameter_keys:
            values = [entry.model_state.get(key) for entry in result.history]
            if all(isinstance(v, (int, float)) for v in values):
                param_ax.plot(iterations, values, marker="o", label=key)
        param_ax.set_title("Parameter trajectory")
        param_ax.set_xlabel("iteration")
        param_ax.grid(True, alpha=0.3)
        if param_ax.get_legend_handles_labels()[0]:
            param_ax.legend()
    else:
        score_ax.set_xlabel("iteration")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def save_html_report(result: RunResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "iteration": entry.iteration,
            "score": entry.score,
            "suggestion": entry.suggestion,
            "model_state": entry.model_state,
            "metrics": entry.metrics,
        }
        for entry in result.history
    ]
    payload = json.dumps(data)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(result.task_name)} experiment report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    canvas {{ border: 1px solid #ddd; max-width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f8f8f8; }}
  </style>
</head>
<body>
  <h1>Experiment Report: {escape(result.task_name)}</h1>
  <p>Best score: <strong>{result.best_score:.6f}</strong></p>
  <canvas id="scoreChart" width="960" height="320"></canvas>
  <table id="historyTable">
    <thead>
      <tr><th>Iteration</th><th>Score</th><th>Suggestion</th><th>Model State</th><th>Metrics</th></tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    const rows = {payload};
    const canvas = document.getElementById('scoreChart');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const padding = 40;
    const xs = rows.map(r => r.iteration);
    const ys = rows.map(r => r.score);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const yRange = (maxY - minY) || 1;
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, h - padding);
    ctx.lineTo(w - padding, h - padding);
    ctx.strokeStyle = '#555';
    ctx.stroke();
    rows.forEach((row, i) => {{
      const x = padding + (i / Math.max(rows.length - 1, 1)) * (w - 2 * padding);
      const y = h - padding - ((row.score - minY) / yRange) * (h - 2 * padding);
      if (i === 0) ctx.beginPath(), ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      ctx.fillStyle = '#1f77b4';
      ctx.fillRect(x - 2, y - 2, 4, 4);
    }});
    ctx.strokeStyle = '#1f77b4';
    ctx.stroke();
    const tbody = document.querySelector('#historyTable tbody');
    rows.forEach(row => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{row.iteration}}</td><td>${{row.score.toFixed(6)}}</td><td>${{row.suggestion}}</td><td>${{JSON.stringify(row.model_state)}}</td><td>${{JSON.stringify(row.metrics)}}</td>`;
      tbody.appendChild(tr);
    }});
  </script>
</body>
</html>
"""
    target.write_text(html, encoding="utf-8")


def plot_mutation_run(
    result: MutationRunResult,
    *,
    show: bool = True,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc
    iterations = [entry.iteration for entry in result.history]
    scores = [entry.score if entry.score is not None else float("nan") for entry in result.history]
    statuses = [entry.status.value for entry in result.history]
    wall_times = [
        float(entry.resource_metrics.get("wall_time_seconds", 0.0))
        for entry in result.history
    ]
    cumulative_wall = []
    total = 0.0
    for value in wall_times:
        total += value
        cumulative_wall.append(total)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    score_ax, status_ax, resource_ax = axes
    score_ax.plot(iterations, scores, marker="o", label="frontier score candidates")
    score_ax.set_ylabel("score")
    score_ax.grid(True, alpha=0.3)
    status_map = {"keep": 2, "discard": 1, "crash": 0}
    numeric_status = [status_map.get(status, -1) for status in statuses]
    status_ax.step(iterations, numeric_status, where="mid")
    status_ax.set_yticks([0, 1, 2], labels=["crash", "discard", "keep"])
    status_ax.set_ylabel("status")
    status_ax.grid(True, alpha=0.3)
    resource_ax.plot(iterations, cumulative_wall, marker="o", color="#9467bd")
    resource_ax.set_ylabel("cum. wall seconds")
    resource_ax.set_xlabel("experiment iteration")
    resource_ax.grid(True, alpha=0.3)
    fig.suptitle(f"{result.task_name}: mutation experiment lifecycle")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
