# Speaker Notes: Beyond Vibe Coding: Building a Ratcheting Experiment Harness in Python
**Conference:** PyCon Hong Kong 2026
**Format:** 15-Minute Short Talk

---

## Presentation Timing Guide
* **0:00 - 1:45 (Slide 1):** Introduction & Core Pitch
* **1:45 - 3:30 (Slide 2):** Framing: The Abstraction Shift
* **3:30 - 6:00 (Slide 3):** Method: Harness Design (How to prevent cheating & noise)
* **6:00 - 8:30 (Slide 4):** Orchestration: Implementation (Python Primitives & Git)
* **8:30 - 11:00 (Slide 5):** The Task: What is the Restaurant Policy? (Deep-Dive)
* **11:00 - 12:15 (Slide 6):** Optimization Ledger (Results)
* **12:15 - 13:30 (Slide 7 & 8):** Progress Graph & Live Replay
* **13:30 - 15:00 (Slide 9):** Key Engineering Takeaways & Q&A

---

## Slide 1: Beyond Vibe Coding · PyCon HK 2026
### Talking Points
* **Opening hook:** How many of you have sat in front of an LLM coding assistant, pasting snippets, prompt-guessing, and trying to check if the code works by just "vibing" it? We call this **Vibe Coding**.
* It’s fast for writing single methods. But it breaks down completely when you want to build robust, reproducible systems.
* Today, we are discussing the Python engineering architecture behind shifting from *chatting with code* to running programmatically-orchestrated *sandboxed experiments* that strictly **ratchet** forward only when there is a scalar performance improvement.

---

## Slide 2: Shift the Abstraction Layer
### Talking Points
* Let's layout the hierarchy of AI-assisted engineering:
  1. **Vibe Coding:** Manual copy-pasting. Zero metrics, zero telemetry, full of regression risk.
  2. **Agentic Engineering:** We give an agent terminal/shell privileges. But unconstrained agents often write random scripts, touch files across multiple directories, pollute states, and can break local environments.
  3. **Auto-Research Sandbox:** We lock down the environment. The LLM can only touch **exactly one module file**. We run automated unit tests and deterministic scoring pipelines, recording every single attempt on an immutable ledger.

---

## Slide 3: Harness Design: Preventing Cheating & Noise
### Talking Points
* **The Danger:** If your AI has access to the full repository, it is lazy and clever—it will rewrite the tests to pass, or alter the simulator parameters to award itself a high score!
* **The Solution:** We establish a strict boundary between immutable and mutable codebase boundaries.

### Technical Deep-Dive & Codebase Pointers
1. **Deterministic Pinning:**
   * Why it matters: If the environment features any randomized drift, the AI might submit a bad policy that scores high simply due to a "lucky" random draw. 
   * Where in the code: Look at [autoresearch/tasks.py](autoresearch/tasks.py#L75-L88).
   * Context: Here, we pin isolated training seed constants (`training_seeds`) and validation seed constants (`validation_seeds`) during scenario generation. The scenarios are generated via `_generate_scenario` utilizing deterministic Gaussians using `random.Random(seed)` at [autoresearch/tasks.py](autoresearch/tasks.py#L191). This guarantees that every trial faces the *exact same* sequence of consumer demands.
2. **Subprocess Hygiene:**
   * Why it matters: Modifying, training, and running dynamic candidates sequentially inside a single, stateful continuous Python process causes massive problems. It pollutes `sys.modules` caching, leaks memory across trials, and retains global variable states.
   * Where in the code: Look at [autoresearch/experiments/restaurant_eval.py](autoresearch/experiments/restaurant_eval.py#L90-L98).
   * Context: The harness invokes the evaluator as a separate subprocess within a Docker container. Every single evaluation task starts completely fresh, assuring total execution hygiene.
3. **The Git-Diff Security Guard:**
   * Why it matters: To guarantee that the LLM has only altered the allowed file, we programmatically audit modified files.
   * Where in the code: Look at the helper `_validate_mutable_change_set` in [autoresearch/control_plane.py](autoresearch/control_plane.py#L186-L197).
   * Context: We use Python’s `subprocess` to execute `git diff --name-only`. If the output list contains any modifications beyond the allocated file (`restaurant_train.py`), the run is immediately aborted as illegal.
4. **Test Pre-checking:**
   * Where in the code: Look at [tests/test_autoresearch.py](tests/test_autoresearch.py#L1).
   * Context: Before wasting compute running a full multi-day simulation, standard `unittest` cases run to prove that the candidate's custom `build_policy()` is structurally sound and satisfies interface duck-typing.

---

## Slide 4: Dynamic Imports and Git-backed Atomic Ratchets
### Talking Points
* Now let’s talk about how the harness actually hot-swaps active python files on the fly and maintains history. No complicated databases required—just standard library Python and Git!

### Technical Deep-Dive & Codebase Pointers
1. **Dynamic Custom Imports:**
   * How we load changing code at runtime: Look at `_load_experiment_module` in [autoresearch/experiments/restaurant_eval.py](autoresearch/experiments/restaurant_eval.py#L13-L21).
   * Context: We bypass standard library `import` which locks module caches. Instead, we use `importlib.util.spec_from_file_location("mutation_experiment", target)` and `module_from_spec`. This allows Python to load and execute the code at the given file path as a completely fresh module instance.
2. **The Duck-Typed Policy Contract:**
   * To keep the environment generic, we programmatically implement a strict contract using Python's duck-typing protocols.
   * Look at `evaluate_experiment` in [autoresearch/experiments/restaurant_eval.py](autoresearch/experiments/restaurant_eval.py#L24-L39).
   * Context: The loaded module must expose a single hook `build_policy()`, which returns an object offering `decide_orders(observation)`. If the optional `fit` routine is available, the harness calls `fit(scenarios, task)` prior to simulation.
3. **The Atomic Git Ratchet Loop:**
   * How do we rollback bad changes and keep the good ones without database state? Standard Git commits!
   * Look at [autoresearch/frontier.py](autoresearch/frontier.py#L50-L58).
   * Context: 
     * Prior to evaluating, we automatically commit the active files using `commit_before_run`.
     * If the candidate score underperforms the previous record (obtained by reading `results.tsv` via `read_best_result`), we call `revert_last_commit`, executing a hard reset: `git reset --hard HEAD~1`.
     * If the score improves, we record the candidate's SHA, timestamp, and score as `decision="keep"` inside `results.tsv` (look at [autoresearch/frontier.py](autoresearch/frontier.py#L71-L93)).

---

## Slide 5: The Task: What is the Restaurant Policy? (Deep-Dive)
### Talking Points
* Before we talk about the results, let us understand what the agent is actually working on and changing under the hood. What is this restaurant simulation and what is the policy code?
* The task is a daily restaurant inventory simulation model where raw ingredients must be ordered in the face of overlapping lead days, perishability, and shared storage boundaries. Let us inspect the two primary policies implemented in [autoresearch/experiments/restaurant_train.py](autoresearch/experiments/restaurant_train.py): the heuristic baseline and our machine-learning optimized variant.

### Technical Deep-Dive & Codebase Pointers
1. **The Heuristic: AdaptiveRestaurantPolicy**
   * **Demand Blending:** To predict overnight ingredient demand, the heuristic first fits the data by looking up average weekday-specific and global usage profiles from historical training frames. On any given simulated day, it blends the recent 5-day moving average with the long-term historical base demand using a parameter `recent_demand_weight`.
   * **Target Stock Calculations:** The logic estimates the necessary stocking level to survive until the next delivery arrives. It uses the equation:
     $$\text{Target Stock} = \text{Average Daily Demand} \times \text{Coverage Days} + \text{Safety Factor} \times \sqrt{\text{Average Daily Demand}}$$
     where $\text{Coverage Days} = \text{Lead Time Days} + 1$.
   * **Freshness & Spoilage Protection:** To handle perishable ingredients (like lettuce with short shelf-lives), it multiplies the target down if inventory is highly perishable or applies a specialized `freshness_bias` factor to avoid over-ordering stock that will simply decay in the dumpster.
   * **Shared Storage Constrained Budgeting:** In [autoresearch/experiments/restaurant_train.py](autoresearch/experiments/restaurant_train.py#L225-L270), we run a proportional allocator named `_budget_order_requests`. If the sum of currently on-hand stock, incoming pipelines, and newly requested orders exceeds the restaurant's global physical storage capacity limit, ingredients are ranked by their priority multiplier. The requested order volumes are then proportionally scaled back so that the physical warehouse limits are never breached, preventing expensive offlink storage penalty costs.

2. **The Machine Learning Model: FocusResidualHybridPolicy**
   * **Perfect-Knowledge Oracle Training:** We compile idealized target labels using a hindsight oracle. By inspecting the actual upcoming demand sequences in future training frames, `_oracle_orders_for_day` computes the exact optimal volumes needed to make zero stockouts while perfectly obeying warehouse capacity.
   * **Feature Extraction:** For each simulated day, we construct feature arrays including cyclical weekday triggers (represented as $\sin\left(\frac{2\pi \cdot \text{day}}{7}\right)$ and $\cos\left(\frac{2\pi \cdot \text{day}}{7}\right)$), currently available stocks, and quantities traveling in the delivery pipeline.
   * **MLP Regression training:** In [autoresearch/experiments/restaurant_train.py](autoresearch/experiments/restaurant_train.py#L905-L935), we normalize features using `StandardScaler` from scikit-learn, then train a multi-layer perceptron regressor (`MLPRegressor` with 48x24 hidden layers) using Adam optimization.
   * **Focused Residual Overlay (The Best):** Rather than letting a neural network predict inventory decisions from scratch (which risks catastrophic out-of-distribution pipeline crashes), the optimized candidate operates as a focused hybrid. It runs the robust heuristic first, then uses the MLP model to predict positive residual tweaks only on the volatile key ingredients: cheese, onion, lettuce, and pasta. This preserves rule-based safety features while overlays learn subtle demand patterns to fine-tune holding costs.

---

## Slide 6: Recorded Python Optimization Ledger
### Talking Points
* Show the actual numbers. The benchmark shifts from a starting score of negative **-28,735.04** (where we lose almost $28k on tomatoes and storage violations) up to positive **$3,866.29**.
* Point out that this is recorded automatically in the ledger file [results.tsv](results.tsv).

---

## Slide 7 & 8: Progress Graph & Live Replay
### Talking Points
* Display the step-by-step progress points.
* Emphasize: Point out how each improvement correlates with a specific git commit message on Slide 7.
* Present Slide 8: It embeds the interactive simulation replay telemetry, generated cleanly through [autoresearch/reporting.py](autoresearch/reporting.py) during execution.

---

## Slide 9: Practical Python Engineering Lessons
### Talking Points
* Wrap up with the real execution takeaways for any python engineer:
  1. **Avert Process Pollution:** Never dynamically swap code inside a single persistent process. Separate them using Docker, Git Worktrees, or subprocess containers.
  2. **NVIDIA Driver Mismatches:** If GPU runtime triggers host system mismatches, keep robust fallback code paths (e.g. CPU or plain command fallbacks) to prevent pipeline blocks.
  3. **Autonomy needs Boundaries:** AI operates best when given a strict contract. Define an explicit interface, validate non-policy changes with `git diff`, and let Python programmatically orchestrate the search space.
