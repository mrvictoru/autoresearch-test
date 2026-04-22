from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_report_bundle(output_dir: str | Path, artifact: dict[str, Any]) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "run_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    report_path = target_dir / "report.html"
    report_path.write_text(_render_report_html(artifact), encoding="utf-8")
    return target_dir


def _render_report_html(artifact: dict[str, Any]) -> str:
    payload = json.dumps(artifact, separators=(",", ":"))
    template = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Restaurant Run Report</title>
  <style>
    :root {
      --bg: #0c1018;
      --panel: #151b27;
      --panel-2: #101621;
      --line: #2c3649;
      --text: #edf1f7;
      --muted: #9aa6bc;
      --accent: #f0b34f;
      --accent-2: #5ed09a;
      --danger: #e36a72;
      --warn: #f3b35f;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: radial-gradient(circle at top, #151b27 0%, var(--bg) 55%); color: var(--text); }
    .page { max-width: 1440px; margin: 0 auto; padding: 24px; }
    h1, h2, h3, p { margin: 0; }
    .hero { display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 24px; }
    .hero-copy p { color: var(--muted); margin-top: 8px; max-width: 820px; line-height: 1.5; }
    .hero-meta { display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 10px; min-width: 280px; }
    .metric-card, .panel { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0)); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 40px rgba(0,0,0,0.24); }
    .metric-card { padding: 14px 16px; }
    .metric-card .label { color: var(--muted); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-card .value { margin-top: 6px; font-size: 24px; color: var(--accent); }
    .summary-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 20px; }
    .controls { display: grid; grid-template-columns: 240px 1fr auto auto; gap: 12px; padding: 18px; align-items: center; margin-bottom: 20px; }
    select, button, input[type=range] { width: 100%; }
    select, button { background: var(--panel-2); border: 1px solid var(--line); color: var(--text); border-radius: 12px; padding: 12px; font: inherit; }
    button { cursor: pointer; width: auto; min-width: 100px; }
    .viewer-grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; margin-bottom: 20px; }
    .panel { padding: 18px; }
    .panel h2 { margin-bottom: 14px; font-size: 16px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
    .panel-caption { color: var(--muted); font-size: 12px; line-height: 1.5; margin: -6px 0 14px; }
    .event-headline { font-size: 20px; margin-bottom: 12px; }
    .event-meta { color: var(--muted); font-size: 13px; display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
    .flow-layout { position: relative; display: grid; grid-template-columns: minmax(180px, 0.9fr) minmax(260px, 1.1fr) minmax(220px, 1fr); gap: 18px; margin-top: 18px; min-height: 520px; }
    .flow-column { position: relative; z-index: 1; display: grid; gap: 12px; align-content: start; }
    .flow-column-title { color: var(--muted); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 2px; }
    .flow-spacer { position: relative; z-index: 1; }
    .flow-links { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; }
    .flow-links path { fill: none; stroke: rgba(111, 128, 170, 0.18); stroke-width: 2.25; stroke-linecap: round; }
    .flow-links path.active { stroke: rgba(240, 179, 79, 0.78); filter: drop-shadow(0 0 7px rgba(240, 179, 79, 0.24)); }
    .flow-links path.success { stroke: rgba(94, 208, 154, 0.82); filter: drop-shadow(0 0 8px rgba(94, 208, 154, 0.28)); }
    .flow-links path.failure { stroke: rgba(227, 106, 114, 0.86); filter: drop-shadow(0 0 8px rgba(227, 106, 114, 0.28)); }
    .flow-node { position: relative; border-radius: 18px; border: 1px solid rgba(66, 79, 105, 0.72); background: linear-gradient(180deg, rgba(15,21,34,0.94), rgba(11,16,26,0.92)); padding: 12px 14px; min-height: 76px; display: grid; gap: 8px; transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease, opacity 160ms ease; }
    .flow-node::before { content: ''; position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 999px; border: 2px solid rgba(123, 138, 171, 0.5); background: #0f1521; transform: translateY(-50%); }
    .flow-node.order::before { right: -7px; }
    .flow-node.ingredient::before { left: -7px; }
    .flow-node.idle { opacity: 0.48; }
    .flow-node.active { opacity: 1; transform: translateY(-1px); box-shadow: 0 0 0 1px rgba(240, 179, 79, 0.16), 0 14px 28px rgba(0,0,0,0.28); }
    .flow-node.success { border-color: rgba(94, 208, 154, 0.9); }
    .flow-node.failure { border-color: rgba(227, 106, 114, 0.96); }
    .flow-node.order.success { background: linear-gradient(180deg, rgba(16,44,36,0.88), rgba(11,21,26,0.92)); box-shadow: 0 0 0 1px rgba(94, 208, 154, 0.2), 0 0 18px rgba(94, 208, 154, 0.18); }
    .flow-node.order.failure { background: linear-gradient(180deg, rgba(53,18,27,0.88), rgba(17,14,20,0.92)); box-shadow: 0 0 0 1px rgba(227, 106, 114, 0.2), 0 0 18px rgba(227, 106, 114, 0.18); }
    .flow-node.ingredient.good::before { border-color: rgba(94, 208, 154, 0.92); box-shadow: 0 0 10px rgba(94, 208, 154, 0.4); }
    .flow-node.ingredient.warn::before { border-color: rgba(243, 179, 95, 0.92); box-shadow: 0 0 10px rgba(243, 179, 95, 0.36); }
    .flow-node.ingredient.danger::before { border-color: rgba(227, 106, 114, 0.96); box-shadow: 0 0 10px rgba(227, 106, 114, 0.42); }
    .flow-node.ingredient.success { background: linear-gradient(180deg, rgba(19,41,36,0.84), rgba(11,20,26,0.94)); }
    .flow-node.ingredient.failure { background: linear-gradient(180deg, rgba(48,18,25,0.86), rgba(16,14,21,0.94)); }
    .node-topline { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
    .node-label { font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; }
    .node-metric { color: var(--muted); font-size: 12px; }
    .node-subtext { color: var(--muted); font-size: 12px; line-height: 1.35; min-height: 16px; }
    .inventory-meter { display: grid; gap: 7px; }
    .inventory-meter-row { display: flex; justify-content: space-between; gap: 10px; font-size: 11px; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
    .inventory-track { position: relative; height: 10px; background: #1d2534; border-radius: 999px; overflow: hidden; }
    .inventory-fill { position: absolute; inset: 0 auto 0 0; border-radius: inherit; }
    .inventory-fill.good { background: linear-gradient(90deg, #78e3b4, #5ed09a); }
    .inventory-fill.warn { background: linear-gradient(90deg, #f4cd67, #f0b34f); }
    .inventory-fill.danger { background: linear-gradient(90deg, #f0828a, #e36a72); }
    .ingredient-usage { display: flex; justify-content: space-between; gap: 10px; align-items: center; font-size: 12px; }
    .ingredient-usage .delta { color: var(--muted); }
    .ingredient-usage .delta.critical { color: var(--danger); }
    .ingredient-usage .delta.positive { color: var(--accent-2); }
    .flow-empty { display: grid; place-items: center; min-height: 520px; color: var(--muted); border: 1px dashed rgba(66, 79, 105, 0.6); border-radius: 18px; background: rgba(255,255,255,0.02); }
    .feed { display: grid; gap: 10px; max-height: 520px; overflow: auto; }
    .feed-item { padding: 12px; border-radius: 12px; border: 1px solid var(--line); background: rgba(255,255,255,0.02); }
    .feed-item.active { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(240,179,79,0.25); }
    .feed-item .type { color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .feed-item .desc { margin-top: 6px; color: var(--text); line-height: 1.4; }
    .feed-item .cash.negative { color: var(--danger); }
    .feed-item .cash.positive { color: var(--accent-2); }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    .chart-wrap svg { width: 100%; height: 260px; display: block; background: rgba(0,0,0,0.15); border-radius: 14px; }
    .chart-controls { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
    .tables { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; }
    th { color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px; }
    .pill { display: inline-flex; padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 12px; }
    @media (max-width: 1080px) {
      .summary-grid, .charts, .tables, .viewer-grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
      .hero { flex-direction: column; align-items: stretch; }
    }
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <div class=\"hero-copy\">
        <span class=\"pill\">Restaurant Trace Viewer</span>
        <h1 style=\"margin-top:12px;font-size:34px;\">Business replay and operating report</h1>
        <p>Replay order-by-order operations, inspect ingredient consumption and stockouts, and review cash flow and inventory behavior from the saved simulation trace.</p>
      </div>
      <div id=\"hero-meta\" class=\"hero-meta\"></div>
    </section>
    <section id=\"summary\" class=\"summary-grid\"></section>
    <section class=\"panel controls\">
      <select id=\"scenario-select\"></select>
      <input id=\"event-slider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" />
      <button id=\"play-toggle\">Play</button>
      <button id=\"step-button\">Step</button>
    </section>
    <section class=\"viewer-grid\">
      <div class=\"panel\">
        <h2>Operations replay</h2>
        <p class=\"panel-caption\">Current order is shown on the left, its ingredient dependencies are linked across the panel, and each ingredient exposes live inventory health.</p>
        <div id=\"event-headline\" class=\"event-headline\"></div>
        <div id=\"event-meta\" class=\"event-meta\"></div>
        <div id=\"event-details\" style=\"color:var(--muted);line-height:1.5;\"></div>
        <div id=\"flow-visual\" class=\"flow-layout\"></div>
      </div>
      <div class=\"panel\">
        <h2>Event feed</h2>
        <div id=\"event-feed\" class=\"feed\"></div>
      </div>
    </section>
    <section class=\"charts\">
      <div class=\"panel chart-wrap\">
        <h2>Cash flow</h2>
        <svg id=\"cash-chart\" viewBox=\"0 0 640 260\" preserveAspectRatio=\"none\"></svg>
      </div>
      <div class=\"panel chart-wrap\">
        <div class=\"chart-controls\">
          <h2 style=\"margin:0;flex:1;\">Inventory history</h2>
          <select id=\"ingredient-select\" style=\"max-width:220px;\"></select>
        </div>
        <svg id=\"inventory-chart\" viewBox=\"0 0 640 260\" preserveAspectRatio=\"none\"></svg>
      </div>
    </section>
    <section class=\"tables\">
      <div class=\"panel\">
        <h2>Menu economics</h2>
        <table>
          <thead><tr><th>Item</th><th>Fulfilled</th><th>Lost</th><th>Revenue</th></tr></thead>
          <tbody id=\"menu-table\"></tbody>
        </table>
      </div>
      <div class=\"panel\">
        <h2>Ingredient operations</h2>
        <table>
          <thead><tr><th>Ingredient</th><th>Consumed</th><th>Spoiled</th><th>Missed</th></tr></thead>
          <tbody id=\"ingredient-table\"></tbody>
        </table>
      </div>
    </section>
  </div>
  <script id=\"artifact-data\" type=\"application/json\">__PAYLOAD__</script>
  <script>
    const artifact = JSON.parse(document.getElementById('artifact-data').textContent);
    const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
    const compact = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
    const aggregate = artifact.aggregate_metrics;
    const scenarios = artifact.scenarios;
    const scenarioSelect = document.getElementById('scenario-select');
    const slider = document.getElementById('event-slider');
    const playToggle = document.getElementById('play-toggle');
    const stepButton = document.getElementById('step-button');
    const ingredientSelect = document.getElementById('ingredient-select');
    const heroMeta = document.getElementById('hero-meta');
    const summary = document.getElementById('summary');
    const flowVisual = document.getElementById('flow-visual');
    const eventFeed = document.getElementById('event-feed');
    const eventHeadline = document.getElementById('event-headline');
    const eventMeta = document.getElementById('event-meta');
    const eventDetails = document.getElementById('event-details');
    const cashChart = document.getElementById('cash-chart');
    const inventoryChart = document.getElementById('inventory-chart');
    const menuTable = document.getElementById('menu-table');
    const ingredientTable = document.getElementById('ingredient-table');
    let scenarioIndex = 0;
    let eventIndex = 0;
    let playing = false;
    let timer = null;

    function prettyName(name) {
      return String(name || '').replaceAll('_', ' ');
    }

    function ingredientUniverse(scenario) {
      const names = new Set(Object.keys(scenario.final_inventory || {}));
      for (const checkpoint of scenario.checkpoints || []) {
        for (const name of Object.keys(checkpoint.inventory || {})) {
          names.add(name);
        }
      }
      for (const event of scenario.events || []) {
        for (const name of Object.keys(event.inventory_before || {})) {
          names.add(name);
        }
        for (const name of Object.keys(event.inventory_after || {})) {
          names.add(name);
        }
        for (const name of Object.keys(event.details?.consumed_ingredients || {})) {
          names.add(name);
        }
        for (const name of Object.keys(event.details?.missing_ingredients || {})) {
          names.add(name);
        }
      }
      return [...names].sort((left, right) => prettyName(left).localeCompare(prettyName(right)));
    }

    function menuUniverse(scenario) {
      const names = new Set((artifact.benchmark && artifact.benchmark.menu_items) || []);
      for (const event of scenario.events || []) {
        if (event.event_type === 'customer_order' && event.details?.item_name) {
          names.add(event.details.item_name);
        }
      }
      return [...names].sort((left, right) => prettyName(left).localeCompare(prettyName(right)));
    }

    function computeIngredientCapacityMap(scenario) {
      const capacity = {};
      for (const name of ingredientUniverse(scenario)) {
        capacity[name] = 1;
      }
      for (const checkpoint of scenario.checkpoints || []) {
        for (const [name, value] of Object.entries(checkpoint.inventory || {})) {
          capacity[name] = Math.max(capacity[name] || 1, value || 0);
        }
      }
      for (const event of scenario.events || []) {
        for (const [name, value] of Object.entries(event.inventory_before || {})) {
          capacity[name] = Math.max(capacity[name] || 1, value || 0);
        }
        for (const [name, value] of Object.entries(event.inventory_after || {})) {
          capacity[name] = Math.max(capacity[name] || 1, value || 0);
        }
      }
      return capacity;
    }

    function classifyInventoryLevel(quantity, capacity) {
      if (quantity <= 0) {
        return { tone: 'danger', status: 'Out of stock', ratio: 0 };
      }
      const ratio = capacity > 0 ? quantity / capacity : 0;
      if (ratio <= 0.2) {
        return { tone: 'danger', status: 'Critical', ratio };
      }
      if (ratio <= 0.45) {
        return { tone: 'warn', status: 'Low', ratio };
      }
      return { tone: 'good', status: 'Healthy', ratio };
    }

    function eventOrderState(event) {
      if (!event || event.event_type !== 'customer_order') {
        return null;
      }
      return event.details?.status === 'fulfilled' ? 'success' : 'failure';
    }

    function metricCard(label, value) {
      return `<div class=\"metric-card\"><div class=\"label\">${label}</div><div class=\"value\">${value}</div></div>`;
    }

    function setStaticMeta() {
      heroMeta.innerHTML = [
        metricCard('Policy', artifact.run.policy_name || 'Unknown'),
        metricCard('Seed', compact.format(artifact.run.seed)),
        metricCard('Days', compact.format(artifact.run.days)),
        metricCard('Scenarios', compact.format(scenarios.length)),
      ].join('');
      summary.innerHTML = [
        metricCard('Score', compact.format(aggregate.score)),
        metricCard('Service level', `${compact.format(aggregate.service_level * 100)}%`),
        metricCard('Revenue', currency.format(aggregate.revenue)),
        metricCard('Order cost', currency.format(aggregate.order_cost)),
        metricCard('Stockout penalty', currency.format(aggregate.stockout_penalty)),
      ].join('');
    }

    function currentScenario() {
      return scenarios[scenarioIndex];
    }

    function describeEvent(event) {
      const d = event.details || {};
      if (event.event_type === 'customer_order') {
        return d.status === 'fulfilled'
          ? `${prettyName(d.item_name)} sold; consumed ${Object.entries(d.consumed_ingredients || {}).map(([k,v]) => `${prettyName(k)} x${v}`).join(', ')}`
          : `${prettyName(d.item_name)} lost; missing ${Object.entries(d.missing_ingredients || {}).map(([k,v]) => `${prettyName(k)} x${v}`).join(', ')}`;
      }
      if (event.event_type === 'restock_order') {
        return Object.entries(d.orders || {}).map(([k,v]) => `${prettyName(k)} +${v.quantity} (D${v.arrival_day})`).join(', ');
      }
      if (event.event_type === 'restock_arrival') {
        return Object.entries(d.quantities || {}).map(([k,v]) => `${prettyName(k)} +${v}`).join(', ');
      }
      if (event.event_type === 'spoilage') {
        return Object.entries(d.expired_by_ingredient || {}).map(([k,v]) => `${prettyName(k)} -${v}`).join(', ');
      }
      if (event.event_type === 'holding_cost') {
        return `Holding cost ${currency.format(d.holding_cost || 0)}`;
      }
      return JSON.stringify(d);
    }

    function updateControls() {
      const scenario = currentScenario();
      slider.max = Math.max(0, scenario.events.length - 1);
      slider.value = eventIndex;
      ingredientSelect.innerHTML = ingredientUniverse(scenario).map(name => `<option value=\"${name}\">${prettyName(name)}</option>`).join('');
    }

    function renderFlowVisual(scenario, event) {
      if (!event) {
        flowVisual.innerHTML = '<div class=\"flow-empty\">No event data available for this scenario.</div>';
        return;
      }
      const snapshot = event.inventory_after || event.inventory_before || scenario.final_inventory || {};
      const menuItems = menuUniverse(scenario);
      const capacityMap = computeIngredientCapacityMap(scenario);
      const activeOrder = event.event_type === 'customer_order' ? event.details?.item_name : null;
      const orderState = eventOrderState(event);
      const activeIngredients = new Set([
        ...Object.keys(event.details?.consumed_ingredients || {}),
        ...Object.keys(event.details?.missing_ingredients || {}),
      ]);
      const ingredientNames = ingredientUniverse(scenario).sort((left, right) => {
        const activeDelta = Number(activeIngredients.has(right)) - Number(activeIngredients.has(left));
        if (activeDelta !== 0) {
          return activeDelta;
        }
        return (snapshot[right] || 0) - (snapshot[left] || 0);
      });

      const orderNodes = menuItems.map((name) => {
        const isActive = activeOrder === name;
        const classes = ['flow-node', 'order'];
        if (!isActive) {
          classes.push('idle');
        }
        if (isActive) {
          classes.push('active');
          if (orderState) {
            classes.push(orderState);
          }
        }
        const statusText = !isActive ? 'Waiting' : orderState === 'success' ? 'Fulfilled' : 'Lost sale';
        const recipeText = isActive
          ? Object.entries(event.details?.consumed_ingredients || event.details?.missing_ingredients || {}).map(([ingredient, quantity]) => `${prettyName(ingredient)} x${quantity}`).join(', ')
          : 'Not active in current step';
        return `<div class=\"${classes.join(' ')}\" data-node-kind=\"order\" data-node-name=\"${name}\"><div class=\"node-topline\"><div class=\"node-label\">${prettyName(name)}</div><div class=\"node-metric\">${statusText}</div></div><div class=\"node-subtext\">${recipeText || 'No linked ingredients in this event'}</div></div>`;
      }).join('');

      const ingredientNodes = ingredientNames.map((name) => {
        const quantity = snapshot[name] || 0;
        const capacity = capacityMap[name] || Math.max(quantity, 1);
        const level = classifyInventoryLevel(quantity, capacity);
        const used = (event.details?.consumed_ingredients || {})[name] || 0;
        const missing = (event.details?.missing_ingredients || {})[name] || 0;
        const isActive = activeIngredients.has(name);
        const deltaText = missing > 0 ? `Missing x${missing}` : used > 0 ? `Used x${used}` : quantity <= 0 ? 'Out of stock' : `Buffer ${compact.format(quantity)}`;
        const deltaClass = missing > 0 ? 'critical' : used > 0 ? 'positive' : '';
        const classes = ['flow-node', 'ingredient', level.tone];
        if (!isActive) {
          classes.push('idle');
        }
        if (isActive) {
          classes.push('active');
          classes.push(missing > 0 ? 'failure' : 'success');
        }
        const width = quantity <= 0 ? 0 : Math.max(6, Math.round(level.ratio * 100));
        return `<div class=\"${classes.join(' ')}\" data-node-kind=\"ingredient\" data-node-name=\"${name}\"><div class=\"node-topline\"><div class=\"node-label\">${prettyName(name)}</div><div class=\"node-metric\">${compact.format(quantity)}</div></div><div class=\"inventory-meter\"><div class=\"inventory-meter-row\"><span>${compact.format(quantity)} / ${compact.format(capacity)}</span><span>${level.status}</span></div><div class=\"inventory-track\"><div class=\"inventory-fill ${level.tone}\" style=\"width:${width}%;\"></div></div></div><div class=\"ingredient-usage\"><span class=\"node-subtext\">${isActive ? 'Linked to current order' : 'Standing inventory'}</span><span class=\"delta ${deltaClass}\">${deltaText}</span></div></div>`;
      }).join('');

      flowVisual.innerHTML = `<svg class=\"flow-links\" id=\"flow-links\" viewBox=\"0 0 100 100\" preserveAspectRatio=\"none\"></svg><div class=\"flow-column\"><div class=\"flow-column-title\">Orders</div>${orderNodes}</div><div class=\"flow-spacer\"></div><div class=\"flow-column\"><div class=\"flow-column-title\">Ingredients</div>${ingredientNodes}</div>`;
      drawFlowLinks(event, orderState);
    }

    function drawFlowLinks(event, orderState) {
      const svg = document.getElementById('flow-links');
      if (!svg) {
        return;
      }
      const containerRect = flowVisual.getBoundingClientRect();
      const orderNodes = [...flowVisual.querySelectorAll('[data-node-kind=\"order\"]')];
      const ingredientNodes = [...flowVisual.querySelectorAll('[data-node-kind=\"ingredient\"]')];
      const orderLookup = Object.fromEntries(orderNodes.map((node) => [node.dataset.nodeName, node]));
      const ingredientLookup = Object.fromEntries(ingredientNodes.map((node) => [node.dataset.nodeName, node]));
      const orderName = event?.details?.item_name;
      const relatedIngredients = [
        ...Object.keys(event?.details?.consumed_ingredients || {}),
        ...Object.keys(event?.details?.missing_ingredients || {}),
      ];
      const paths = [];
      const orderNode = orderLookup[orderName];
      if (orderNode) {
        const startRect = orderNode.getBoundingClientRect();
        const startX = startRect.right - containerRect.left;
        const startY = startRect.top - containerRect.top + (startRect.height / 2);
        for (const ingredient of relatedIngredients) {
          const ingredientNode = ingredientLookup[ingredient];
          if (!ingredientNode) {
            continue;
          }
          const endRect = ingredientNode.getBoundingClientRect();
          const endX = endRect.left - containerRect.left;
          const endY = endRect.top - containerRect.top + (endRect.height / 2);
          const curve = Math.max(80, (endX - startX) * 0.45);
          const lineState = (event?.details?.missing_ingredients || {})[ingredient] > 0 ? 'failure' : orderState === 'success' ? 'success' : 'active';
          paths.push(`<path class=\"active ${lineState}\" d=\"M ${startX.toFixed(2)} ${startY.toFixed(2)} C ${(startX + curve).toFixed(2)} ${startY.toFixed(2)}, ${(endX - curve).toFixed(2)} ${endY.toFixed(2)}, ${endX.toFixed(2)} ${endY.toFixed(2)}\" />`);
        }
      }
      svg.setAttribute('viewBox', `0 0 ${Math.max(1, Math.round(containerRect.width))} ${Math.max(1, Math.round(containerRect.height))}`);
      svg.innerHTML = paths.join('');
    }

    function renderEventFeed() {
      const scenario = currentScenario();
      eventFeed.innerHTML = scenario.events.slice(Math.max(0, eventIndex - 8), eventIndex + 9).map((event) => {
        const active = event.event_index === eventIndex ? 'active' : '';
        const cashClass = event.cash_delta < 0 ? 'negative' : 'positive';
        return `<div class=\"feed-item ${active}\"><div class=\"type\">${prettyName(event.event_type)}</div><div class=\"desc\">${describeEvent(event)}</div><div class=\"cash ${cashClass}\">${currency.format(event.cash_delta)}</div></div>`;
      }).join('');
    }

    function buildPath(points, minY, maxY) {
      if (!points.length) return '';
      const width = 620;
      const height = 220;
      const padX = 10;
      const padY = 20;
      const scaleX = points.length > 1 ? (width - padX * 2) / (points.length - 1) : 0;
      const scaleY = maxY === minY ? 0 : (height - padY * 2) / (maxY - minY);
      return points.map((value, index) => {
        const x = padX + index * scaleX;
        const y = height - padY - (scaleY ? (value - minY) * scaleY : height / 2);
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(' ');
    }

    function renderChart(svg, series, color) {
      const minY = Math.min(...series, 0);
      const maxY = Math.max(...series, 1);
      const path = buildPath(series, minY, maxY);
      svg.innerHTML = `<g><line x1=\"0\" y1=\"220\" x2=\"640\" y2=\"220\" stroke=\"#2c3649\" stroke-width=\"1\" /><line x1=\"0\" y1=\"20\" x2=\"640\" y2=\"20\" stroke=\"#2c3649\" stroke-width=\"1\" /><path d=\"${path}\" fill=\"none\" stroke=\"${color}\" stroke-width=\"3\" stroke-linejoin=\"round\" stroke-linecap=\"round\" /></g>`;
    }

    function renderTables() {
      const scenario = currentScenario();
      const byItem = {};
      const byIngredient = {};
      for (const event of scenario.events) {
        if (event.event_type === 'customer_order') {
          const item = event.details.item_name;
          byItem[item] = byItem[item] || { fulfilled: 0, lost: 0, revenue: 0 };
          if (event.details.status === 'fulfilled') {
            byItem[item].fulfilled += 1;
            byItem[item].revenue += event.cash_delta;
            for (const [ingredient, quantity] of Object.entries(event.details.consumed_ingredients || {})) {
              byIngredient[ingredient] = byIngredient[ingredient] || { consumed: 0, spoiled: 0, missed: 0 };
              byIngredient[ingredient].consumed += quantity;
            }
          } else {
            byItem[item].lost += 1;
            for (const [ingredient, quantity] of Object.entries(event.details.missing_ingredients || {})) {
              byIngredient[ingredient] = byIngredient[ingredient] || { consumed: 0, spoiled: 0, missed: 0 };
              byIngredient[ingredient].missed += quantity;
            }
          }
        }
        if (event.event_type === 'spoilage') {
          for (const [ingredient, quantity] of Object.entries(event.details.expired_by_ingredient || {})) {
            byIngredient[ingredient] = byIngredient[ingredient] || { consumed: 0, spoiled: 0, missed: 0 };
            byIngredient[ingredient].spoiled += quantity;
          }
        }
      }
      menuTable.innerHTML = Object.entries(byItem).sort((a, b) => b[1].revenue - a[1].revenue).map(([item, stats]) => `<tr><td>${prettyName(item)}</td><td>${stats.fulfilled}</td><td>${stats.lost}</td><td>${currency.format(stats.revenue)}</td></tr>`).join('');
      ingredientTable.innerHTML = Object.entries(byIngredient).sort((a, b) => b[1].consumed - a[1].consumed).map(([ingredient, stats]) => `<tr><td>${prettyName(ingredient)}</td><td>${stats.consumed}</td><td>${stats.spoiled}</td><td>${stats.missed}</td></tr>`).join('');
    }

    function renderCurrentEvent() {
      const scenario = currentScenario();
      const event = scenario.events[eventIndex];
      eventHeadline.textContent = prettyName(event.event_type);
      eventMeta.innerHTML = [
        `Scenario ${scenario.scenario.name}`,
        `Day ${event.day_index + 1}`,
        event.period ? `Period ${event.period}` : 'Daily event',
        `Event #${event.event_index + 1}`,
        `Cash ${currency.format(event.cumulative_cash)}`,
      ].map(text => `<span>${text}</span>`).join('');
      eventDetails.textContent = describeEvent(event);
      renderFlowVisual(scenario, event);
      renderEventFeed();
    }

    function renderScenario() {
      const scenario = currentScenario();
      eventIndex = Math.min(eventIndex, Math.max(0, scenario.events.length - 1));
      slider.max = Math.max(0, scenario.events.length - 1);
      slider.value = eventIndex;
      renderCurrentEvent();
      renderTables();
      renderChart(cashChart, scenario.events.map(event => event.cumulative_cash), '#5ed09a');
      const selectedIngredient = ingredientSelect.value || ingredientUniverse(scenario)[0];
      const inventorySeries = scenario.checkpoints.filter(checkpoint => checkpoint.inventory && Object.prototype.hasOwnProperty.call(checkpoint.inventory, selectedIngredient)).map(checkpoint => checkpoint.inventory[selectedIngredient]);
      renderChart(inventoryChart, inventorySeries.length ? inventorySeries : [0], '#f0b34f');
    }

    scenarioSelect.innerHTML = scenarios.map((scenario, index) => `<option value=\"${index}\">${scenario.scenario.name}</option>`).join('');
    scenarioSelect.addEventListener('change', () => {
      scenarioIndex = Number(scenarioSelect.value);
      eventIndex = 0;
      updateControls();
      renderScenario();
    });
    slider.addEventListener('input', () => {
      eventIndex = Number(slider.value);
      renderCurrentEvent();
    });
    playToggle.addEventListener('click', () => {
      playing = !playing;
      playToggle.textContent = playing ? 'Pause' : 'Play';
      if (!playing) {
        window.clearInterval(timer);
        timer = null;
        return;
      }
      timer = window.setInterval(() => {
        const max = Number(slider.max);
        eventIndex = eventIndex >= max ? 0 : eventIndex + 1;
        slider.value = eventIndex;
        renderCurrentEvent();
      }, 250);
    });
    stepButton.addEventListener('click', () => {
      const max = Number(slider.max);
      eventIndex = eventIndex >= max ? max : eventIndex + 1;
      slider.value = eventIndex;
      renderCurrentEvent();
    });
    ingredientSelect.addEventListener('change', renderScenario);
    window.addEventListener('resize', () => renderCurrentEvent());

    setStaticMeta();
    updateControls();
    renderScenario();
  </script>
</body>
</html>"""
    return template.replace("__PAYLOAD__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static HTML report from a saved restaurant run artifact.")
    parser.add_argument("artifact", type=str, help="Path to run_artifact.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory; defaults to artifact parent")
    args = parser.parse_args()
    artifact_path = Path(args.artifact)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else artifact_path.parent
    write_report_bundle(output_dir, artifact)


if __name__ == "__main__":
    main()