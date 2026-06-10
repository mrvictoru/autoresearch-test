# Speaker Notes: Build Your Own Ratcheting Experiment Harness
**Conference:** PyCon Hong Kong 2026
**Format:** 15-Minute Short Talk
**Audience:** Python developers who have used LLM coding assistants and want a more rigorous loop.

---

## One-Sentence Mental Model

> *We let the LLM edit one file, score it, and only keep strict improvements. Everything else is plumbing.*

If the audience remembers nothing else, this is the line.

---

## Glossary (define on first use)

- **Ratcheting** — accepting a new candidate only when it strictly beats the best score seen so far. Bad moves are reverted; the frontier only moves up.
- **Mutable / immutable boundary** — the contract that says which files the LLM is allowed to touch and which are frozen. The frozen side is the source of trust.
- **Frontier ledger** — an append-only record of every attempt: timestamp, sha, score, decision (keep / discard / crash), short message. In this repo it is `results.tsv`.
- **Duck-typed contract** — a runtime interface the loaded module must satisfy, e.g. `build_policy()` returns an object with `decide_orders(observation)`. No base class required.
- **Capacity budgeting** — a self-imposed ceiling on what the policy will request so the simulator does not silently clip the most important orders.
- **Residual hybrid** — a learned correction layered on top of a robust rule-based policy. The rule keeps the system safe; the model adds small positive nudges.
- **Plateau exploration** — when 5+ kept attempts have not improved, the harness permits — and encourages — substantially different strategies, with strict reverting.

---

## Presentation Timing Guide (15 minutes total)

| Slide | Title                              | Time      | Speaker seconds |
|-------|------------------------------------|-----------|-----------------|
| 1     | Opening hook                       | 0:00–1:00 | 60              |
| 2     | The mental model                   | 1:00–2:30 | 90              |
| 3     | The loop, end to end               | 2:30–4:00 | 90              |
| 4     | Guardrails vs. threats             | 4:00–5:30 | 90              |
| 5     | The duck-typed policy contract     | 5:30–7:00 | 90              |
| 6     | The ratchet + ledger               | 7:00–8:30 | 90              |
| 7     | The restaurant example             | 8:30–10:00| 90              |
| 8     | Results (real-data curve)          | 10:00–11:30 | 90            |
| 9     | Live replay                        | 11:30–13:30 | 120           |
| 10    | Build your own + Q&A               | 13:30–15:00 | 90            |

Two hard rules:
- Never let slide 7 eat slide 8. If you are running late, cut restaurant deep code — the recipe is in the appendix.
- Always leave 60 seconds for Q&A. If a question runs long, defer to the "Build your own" page where every answer lives.

---

## Slide 1 — Opening hook (0:00–1:00)

**On stage:**
- "Raise your hand if you've pasted a snippet into a chat, eyeballed the output, and shipped it. I have."
- "That works once. It does not survive 200 iterations."
- "Today: a 15-minute pattern that turns 'chat with an LLM' into a strict, reproducible, score-ratcheting experiment loop in Python."

**What not to do:** do not read the slide. Land the hook, then move.

**Transition line:** *"It is one sentence: we let the LLM edit one file, score it, and only keep strict improvements."*

---

## Slide 2 — The mental model (1:00–2:30)

**One diagram, three boxes:**

```
┌──────────────────────┐    ┌──────────────────────┐
│  Immutable simulator │    │  Mutable policy file │
│  (the truth)         │    │  (the LLM's surface)  │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           └─────────►  ratchet  ◄─────┘
                        loop
                          │
                          ▼
                  results.tsv ledger
```

- The **simulator** is the trust anchor. It does not change. It is a function from `policy → score`.
- The **mutable file** is the only thing the LLM is allowed to modify. It must expose a single hook, `build_policy()`.
- The **ratchet loop** runs the policy, reads the score, and only accepts strict improvements. Everything else is reverted.
- The **ledger** is an append-only TSV that records every attempt. It is the audit trail and the place new runs go to read the current best.

**Why this framing matters for newcomers:** every other question in the talk is now anchored. "Why subprocess?" — to keep the simulator trustworthy. "Why git revert?" — because the ratchet only moves up. "Why one file?" — that is the mutable surface.

**Transition line:** *"Now let's open the loop and see what each step actually does."*

---

## Slide 3 — The loop, end to end (2:30–4:00)

**Seven boxes, one column. Walk top-to-bottom, fast.**

1. **Read the best** — open `results.tsv`, parse the highest `keep` row. This is your bar.
2. **Edit the policy** — change only `restaurant_train.py` (or your domain equivalent). One commit, one file.
3. **Pre-check** — run `unittest` so syntax / import / contract errors fail fast and do not waste a full simulation.
4. **Evaluate in a fresh process** — `subprocess` + Docker. No shared `sys.modules`, no leaked globals, no warmed-up caches.
5. **Parse the score** — read `score` from the `--- RESULTS ---` block, never from chat.
6. **Decide** — strictly better than the bar? Keep. Else revert via `git reset --hard HEAD~1`. Crash? Record as `crash`, revert, move on.
7. **Append to the ledger** — `timestamp | branch | sha | score | decision | message`. Always. Even for `discard` and `crash`.

**Closing line on this slide:** *"That's the whole loop. Everything after this is a defense of one of these steps."*

---

## Slide 4 — Guardrails vs. threats (4:00–5:30)

**Single table, scannable in 10 seconds:**

| Threat                              | What it would do                            | Where it is prevented                                                                                       |
|-------------------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Lucky random draw wins the score    | A bad policy passes by noise                | Pinned `training_seeds` / `validation_seeds` in [autoresearch/tasks.py](autoresearch/tasks.py)              |
| LLM edits the evaluator or tests    | Self-graded homework                        | `git diff --name-only` gate in [autoresearch/control_plane.py](autoresearch/control_plane.py)                |
| Stale state across runs             | "Fix" hides a regression from last attempt  | Subprocess + Docker eval in [autoresearch/experiments/restaurant_eval.py](autoresearch/experiments/restaurant_eval.py) |
| Silent regression sneaks in         | Frontier moves sideways or down             | `commit_before_run` + `revert_last_commit` in [autoresearch/frontier.py](autoresearch/frontier.py)          |

**Speaker motion:** read the first column, point to the third column. Do not re-explain the code. The audience will look it up.

**Transition line:** *"Two of those rows depend on a single contract between the harness and the policy file. Let's see the contract."*

---

## Slide 5 — The duck-typed policy contract (5:30–7:00)

**On stage:** show the smallest possible `build_policy()` and `decide_orders()` skeleton. Six lines.

```python
# autoresearch/experiments/restaurant_train.py
def build_policy():
    return MyPolicy(...)

class MyPolicy:
    def fit(self, scenarios, task): ...        # optional
    def decide_orders(self, observation): ...  # required
```

- The contract is **runtime-enforced**, not type-declared. The harness `getattr()`s `build_policy`, calls it, then `hasattr()`s `decide_orders`.
- The contract is **stable across domains**. Restaurant, scheduling, bidding, routing — same shape, different state.
- The contract is **the only thing the LLM is allowed to change**. No imports of internal helpers, no monkey-patching of the simulator.

**Transition line:** *"Now we can talk about the actual ratchet: how a keep or revert happens."*

---

## Slide 6 — The ratchet and the ledger (7:00–8:30)

**Two micro-snippets, six lines each.**

```python
# autoresearch/frontier.py — commit, then maybe revert
commit_before_run("candidate: tweak safety factor")
score = parse_score_from_log()
if score > read_best_result("results.tsv")["score"]:
    append_result(..., decision="keep",  message=msg)
else:
    revert_last_commit()
    append_result(..., decision="discard", message=msg)
```

```text
# results.tsv — one line per attempt, append-only
timestamp  branch                sha     score  decision message
2026-04-30 autoresearch/20260430-start 5c64f4c -28735.04 keep     baseline
2026-04-30 autoresearch/20260430-start 646a3d8  -6157.12 keep     tuned adaptive
2026-04-30 autoresearch/20260430-start 85e83ea   3866.29 keep     service/cashflow hybrid
```

**Three takeaways:**

1. The ledger is the source of truth for "what is the current best." New runs read it before they decide.
2. The git tree is the source of truth for "what code produced this score." `git checkout <sha>` reproduces any past attempt.
3. The two together make the system *auditable* — you can show exactly which file produced exactly which score on exactly which commit.

**Transition line:** *"Let's ground all of that in a concrete example. The restaurant."*

---

## Slide 7 — The restaurant example (8:30–10:00)

**Three bullets. Do not deep-dive the code.**

- **What is the simulator?** A 14-day, two-period-per-day (lunch, dinner) inventory simulation. Menu items share ingredients; ingredients have shelf life, lead time, and per-ingredient storage caps. There is also a global 1000-unit warehouse cap. Score = revenue − waste − holding − order − stockout penalty.
- **What is the policy file?** [autoresearch/experiments/restaurant_train.py](autoresearch/experiments/restaurant_train.py). It exposes `build_policy()`. The current best is a focused residual hybrid: a rule-based heuristic for safety, an MLP residual overlay on cheese / onion / lettuce / pasta.
- **What is the contract?** Same as the slide 5 contract. The restaurant is one instance of the pattern; it is not the pattern.

**If pressed for time, cut this slide first.** The restaurant deep-dive lives in the speaker notes appendix below for the Q&A.

---

## Slide 8 — Results, real-data curve (10:00–11:30)

**The chart on the slide is generated from `results.tsv` at page load.** That means the curve is always live and never lies.

- Open with the headline number: **−28,735 → 3,866**, six kept attempts.
- Walk the chart: each dot is a `keep`. Each label is the commit message. The line is the ratchet — it only goes up.
- Emphasize: **the bar at the start is the failure mode without any of the guardrails.** A loose scoring function would have happily accepted noisy drift and looked like progress.

**Transition line:** *"Let me show you what the simulator actually saw during the best run."*

---

## Slide 9 — Live replay (11:30–13:30)

**Two minutes, mostly hands-off.** The slide is an iframe to [docs/reports/best-20260430-service-cashflow/report.html](docs/reports/best-20260430-service-cashflow/report.html).

- The report is generated by [autoresearch/reporting.py](autoresearch/reporting.py) from a JSON artifact produced by the evaluator.
- Talk through what is on screen: per-day orders, fulfillment, spoilage, cash. The point is **observability after the fact**, which is what makes the loop debuggable.
- If the iframe fails (no network, sandboxed), fall back to a 30-second narrated walk of `run_artifact.json`.

---

## Slide 10 — Build your own + Q&A (13:30–15:00)

**This is the takeaway slide.** Send people to [docs/build_your_own.md](docs/build_your_own.md).

- The five required files: `task.py`, `eval.py`, `train.py`, `frontier.py`, plus the ledger.
- The four required safety checks: fixed seeds, single-file mutation, fresh process, strict better-than-best.
- The five-step domain porting checklist:
  1. Define state / action / constraints.
  2. Write a deterministic scenario generator.
  3. Pick a scalar objective.
  4. Build a baseline policy.
  5. Run the ratchet.
- Common pitfalls: forgetting the subprocess boundary, letting `sys.modules` cache leak, accepting the first improvement instead of strict, conflating "score went up" with "model improved."

**End line:** *"The LLM is not the trick. The boundary is. Define it, enforce it, and the loop does the work."*

---

## Q&A prep (60 seconds)

- **"Why not just use Optuna / Ray Tune?"** — you can, but the boundary and the auditability are what make this safe to run unattended. The optimizer is replaceable; the contract is not.
- **"Why git and not a database?"** — git already gives you content-addressed history, diffing, worktrees, and revert. A database would duplicate all of that. The ledger is the only flat file you actually need.
- **"How do you avoid the LLM cheating by tuning the simulator?"** — the simulator is frozen. The git-diff guard rejects any commit that touches it. The harness refuses to run an unverified worker.
- **"Can I use this on a non-Python repo?"** — yes, in principle. The contract is language-agnostic. The implementation here is Python because the talk is at PyCon.
- **"What's next for the project?"** — multi-worker parallelism via git worktrees, plateau-aware exploration, and a porting guide for non-inventory problems.

---

## Speaker notes appendix — restaurant deep dive (only if asked)

### The heuristic: AdaptiveRestaurantPolicy
- **Demand blending:** weekday-specific average usage vs. last-N-day rolling usage, weighted by `recent_demand_weight`.
- **Target stock:** `avg_daily × (lead_time + 1) + safety_factor × √avg_daily`.
- **Freshness bias:** down-weight ordering for ingredients with very short shelf life, scaled by `freshness_bias`.
- **Capacity budgeting:** in [autoresearch/experiments/restaurant_train.py](autoresearch/experiments/restaurant_train.py) `_budget_order_requests`, when `on_hand + pipeline + requests > global_capacity`, the policy scales requests down proportionally by priority. This is what makes the warehouse behave the way the model expects.

### The residual hybrid: FocusResidualHybridPolicy
- **Oracle labels:** for each training day, compute the exact order volumes that would have produced zero stockouts using the actual future demand of the training scenario. These become supervised targets.
- **Features:** cyclic `sin/cos(2π·day/7)`, on-hand ratio, pipeline ratio, lead time, shelf life, cost signals.
- **Training:** `StandardScaler` + `MLPRegressor(48, 24)` with Adam.
- **Residual:** the model is asked to predict a *positive correction* to the heuristic's order, but only for cheese / onion / lettuce / pasta. The heuristic stays in charge of safety; the model only nudges where it is trustworthy.

### Why "focused" matters
A pure neural policy that predicts orders end-to-end is one bad OOD input away from a catastrophic under-order. Wrapping a robust heuristic and letting the model add a small, gated, positive residual is the actual engineering insight: **the heuristic is the safety net; the model is the dial**.
