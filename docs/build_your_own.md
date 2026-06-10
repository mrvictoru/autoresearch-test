# Build Your Own Ratcheting Experiment Harness

A copy-pasteable recipe for going from "I have a problem and a Python function to score solutions" to "I have a strict, reproducible, score-ratcheting experiment loop in Python that an LLM can drive."

This is the takeaway page for the PyCon HK 2026 talk. The pitch in one sentence:

> *We let the LLM edit one file, score it, and only keep strict improvements. Everything else is plumbing.*

---

## The mental model in one diagram

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

Three components, one loop, one file. That is the whole idea.

---

## The five required files

A working harness is exactly five small files plus one ledger. You can fit them in your head.

| File             | Mutable?  | Purpose                                                              |
|------------------|-----------|----------------------------------------------------------------------|
| `task.py`        | immutable | Defines state, action, constraints, the scenario generator, scoring. |
| `eval.py`        | immutable | Loads `train.py`, runs scenarios, prints stable `--- RESULTS ---`.  |
| `train.py`       | **mutable** | Exposes `build_policy()`. This is the LLM's only surface.          |
| `frontier.py`    | immutable | Commits, reverts, reads/writes the ledger.                           |
| `run_once.sh`    | immutable | One atomic helper: commit → evaluate → keep/discard → log.           |
| `results.tsv`    | append-only ledger of every attempt (one row per attempt).           |

The split is the design. If a file is mutable, treat its contents as untrusted. If a file is immutable, you can re-run the loop against it forever.

---

## The required policy contract

The contract is **runtime-enforced**, not type-declared. The harness looks up the function and the method by name. No base class required.

```python
# train.py — the LLM's only writable surface
def build_policy():
    return MyPolicy(...)

class MyPolicy:
    def fit(self, scenarios, task): ...        # optional
    def decide_orders(self, observation): ...  # required
```

Rules:

- `build_policy()` must exist and return an object. Returning `None` is a crash.
- The returned object must have a `decide_orders(observation)` method. The harness will check it.
- If the policy wants to train on the training scenarios, it may implement `fit(scenarios, task)`. The harness will call it once before validation.
- The contract is stable across domains. Restaurant, scheduling, bidding, routing, prompt-selection, hyperparameter search — same shape, different state.

The harness check is two lines:

```python
build_fn = getattr(module, "build_policy", None)
policy = build_fn()
if not hasattr(policy, "decide_orders"):
    raise RuntimeError("build_policy() must return an object with decide_orders(...)")
```

---

## The four required safety checks

If you skip any one of these, the loop will silently cheat itself.

1. **Fixed seeds.** Pin every random source in your scenario generator and policy training. If seeds drift, the score is noise, not signal.
2. **Single-file mutation.** Before evaluating, run `git diff --name-only` and reject any commit that touched files outside the allowed mutable set. This is the "no editing the tests" guard.
3. **Fresh process.** Run the evaluator in a subprocess (ideally a Docker container). A persistent Python process will leak `sys.modules` caches, warm caches, and global state across runs.
4. **Strict better-than-best.** Read the current best score from the ledger, evaluate the candidate, and accept only if the new score is *strictly* greater. Ties are discards.

These are not optimizations. They are the difference between a ratchet and a random walk.

---

## The required outputs

A run is only useful if downstream tools (humans, agents, charts) can parse it.

- **Stable results block.** `eval.py` must print:

  ```text
  --- RESULTS ---
  score                3866.290000
  service_level          0.812000
  ...
  ```

- **Machine-readable metric line.** Right after the block, print `METRIC_JSON: { ... }`. This is what the harness parses.
- **Append-only ledger rows.** One row per attempt, including `discard` and `crash`. Header:

  ```text
  timestamp  branch  sha  score  decision  message
  ```

- **Optional report bundle.** When `eval.py` is given a `--report-dir`, write `run_artifact.json` plus a self-contained `report.html`. This is for humans; the ratchet does not depend on it.

---

## The five-step domain porting checklist

Use this when you want to apply the pattern to a new problem.

1. **Define state / action / constraints.** What can the policy observe? What can it decide? What hard rules must it never break?
2. **Write a deterministic scenario generator.** Pin seeds, fix any randomness, and split into `training_scenarios()` and `validation_scenarios()`.
3. **Pick a scalar objective.** One number. Higher is better. Everything else is a diagnostic. If you cannot express the goal as one number, you are not ready to ratchet.
4. **Build a baseline policy.** A simple rule that is obviously not optimal. It gives the ratchet a bar to beat and you a reference for "what is a sane output."
5. **Run the ratchet.** `run_once.sh "short message"` in a loop. Read the ledger, not the chat.

---

## Common pitfalls

These are the failure modes we hit most often. Avoid them on the first day.

- **Forgetting the subprocess boundary.** A long-running harness process will eventually produce scores that depend on previous runs' globals. If your scores look "spooky," the boundary is the first thing to check.
- **Letting `sys.modules` cache leak.** When you load a candidate by path, use `importlib.util.spec_from_file_location` so Python does not reuse a cached module from a previous run.
- **Accepting the first improvement instead of strict.** "Looks better" is a recipe for silent regression. The ratchet moves up only on a strict greater-than.
- **Conflating "score went up" with "model improved."** Score can move because of demand variation, seed drift, or a bug. Strict better-than-best on a fixed seed set is the only signal you can trust.
- **Touching the simulator.** The whole point of the boundary is that the simulator is the trusted scorer. If the LLM can edit the scorer, the score is meaningless.
- **Logging only `keep` rows.** `discard` and `crash` are the most informative rows in the ledger. They tell you where the search is failing. Log everything.

---

## A minimal harness in 30 lines

This is a complete, working skeleton. It is intentionally small so you can read it end-to-end before extending it.

```python
# task.py
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class Observation:
    value: int

class MyTask:
    def __init__(self, *, seed: int = 7):
        self.training_seeds = (seed + 1, seed + 2, seed + 3)
        self.validation_seeds = (seed + 100, seed + 200)

    def training_scenarios(self):   return [self._scenario(s) for s in self.training_seeds]
    def validation_scenarios(self): return [self._scenario(s) for s in self.validation_seeds]

    def _scenario(self, seed):
        rng = random.Random(seed)
        return [Observation(value=rng.randint(0, 10)) for _ in range(20)]

    def evaluate(self, policy):
        total = 0.0
        for scenario in self.validation_scenarios():
            for obs in scenario:
                total += float(policy.decide_action(obs).get("x", 0) == obs.value)
        return {"score": total}
```

```python
# train.py
def build_policy():
    class MyPolicy:
        def fit(self, scenarios, task): pass
        def decide_action(self, observation):
            return {"x": observation.value}
    return MyPolicy()
```

```python
# eval.py
import importlib.util, json, sys

def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_policy()

task = __import__("task").MyTask()
policy = load("train.py")
if hasattr(policy, "fit"):
    policy.fit(task.training_scenarios(), task)
metrics = task.evaluate(policy)
print("--- RESULTS ---")
print(f"score {metrics['score']:.6f}")
print(f"METRIC_JSON: {json.dumps(metrics, sort_keys=True)}")
```

```bash
# run_once.sh
set -euo pipefail
MSG="${1:-harness run}"
git add -u && git commit --allow-empty -m "$MSG" || true
python eval.py > run.log 2>&1 || { echo "crash"; exit 1; }
SCORE=$(grep -E '^score ' run.log | awk '{print $2}')
BEST=$(awk -F'\t' '$5=="keep" {print $4}' results.tsv | sort -g | tail -1)
if awk -v a="$SCORE" -v b="$BEST" 'BEGIN{exit !(a+0 > b+0)}'; then
    printf "%s\t%s\tkeep\t%s\n" "$(date -u +%FT%TZ)" "$(git rev-parse --short HEAD)" "$MSG" >> results.tsv
else
    git reset --hard HEAD~1
    printf "%s\t%s\tdiscard\t%s\n" "$(date -u +%FT%TZ)" "$(git rev-parse --short HEAD)" "$MSG" >> results.tsv
fi
```

That is the whole loop. Replace `MyTask` with your problem, replace `MyPolicy` with your baseline, and iterate.

---

## A starter directory layout

```
.
├── task.py
├── eval.py
├── train.py              # the only file the LLM is allowed to edit
├── frontier.py           # commit, revert, ledger helpers
├── run_once.sh
├── results.tsv           # generated on first run
└── tests/
    └── test_contract.py  # asserts build_policy() and decide_action exist
```

Keep it this small until you outgrow it. Most projects never need to.

---

## Where to read next in the autoresearch repo

- [docs/speaker_notes.md](speaker_notes.md) — the talk that motivates this page.
- [docs/components.md](components.md) — what each piece of the autoresearch repo does.
- [docs/guide.md](guide.md) — how the harness is wired up in practice.
- [autoresearch/tasks.py](../autoresearch/tasks.py) — a real (and richer) simulator than the minimal `MyTask` above.
- [autoresearch/experiments/restaurant_eval.py](../autoresearch/experiments/restaurant_eval.py) — a real evaluator with `--- RESULTS ---` and `METRIC_JSON`.
- [autoresearch/frontier.py](../autoresearch/frontier.py) — real commit, revert, and ledger helpers.
- [autoresearch/control_plane.py](../autoresearch/control_plane.py) — the `_validate_mutable_change_set` guard and the multi-worker plumbing.
- [scripts/run_once.sh](../scripts/run_once.sh) — the production version of the 30-line helper above.
