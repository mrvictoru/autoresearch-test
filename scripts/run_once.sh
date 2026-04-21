#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_PATH="${RESULTS_PATH:-$REPO_ROOT/results.tsv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/artifacts/harness}"
MESSAGE="${1:-harness run $(date -u +"%Y-%m-%dT%H:%M:%SZ")}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"

python - "$RESULTS_PATH" <<'PY'
from pathlib import Path
import sys

from autoresearch.frontier import init_results_tsv

init_results_tsv(Path(sys.argv[1]))
PY

BRANCH_NAME="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
BEST_BEFORE="$(
python - "$RESULTS_PATH" <<'PY'
from pathlib import Path
import sys

from autoresearch.frontier import read_best_result

best = read_best_result(Path(sys.argv[1]))
print("" if best is None else best["score"])
PY
)"

CANDIDATE_SHA="$(
python - "$REPO_ROOT" "$MESSAGE" <<'PY'
from pathlib import Path
import sys

from autoresearch.frontier import commit_before_run

print(commit_before_run(sys.argv[2], repo_root=Path(sys.argv[1])))
PY
)"

RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")-${CANDIDATE_SHA:0:12}"
LOG_PATH="$LOG_DIR/run-$RUN_ID.log"

run_evaluator() {
  (
    cd "$REPO_ROOT"
    python -m autoresearch.experiments.restaurant_eval --experiment autoresearch/experiments/restaurant_train.py
  ) >"$LOG_PATH" 2>&1
}

show_log_tail() {
  echo "evaluation failed, last 50 log lines:" >&2
  tail -n 50 "$LOG_PATH" >&2 || true
}

parse_score() {
  python - "$LOG_PATH" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"^\s*score\s+(-?\d+(?:\.\d+)?)", text, flags=re.MULTILINE)
if match is None:
    raise SystemExit(1)
print(match.group(1))
PY
}

record_result() {
  local score_value="$1"
  local decision="$2"
  local summary="$3"
  python - "$RESULTS_PATH" "$BRANCH_NAME" "$CANDIDATE_SHA" "$score_value" "$decision" "$summary" <<'PY'
from pathlib import Path
import sys

from autoresearch.frontier import append_result

score_arg = sys.argv[4]
append_result(
    Path(sys.argv[1]),
    branch=sys.argv[2],
    sha=sys.argv[3],
    score=None if score_arg == "" else float(score_arg),
    decision=sys.argv[5],
    message=sys.argv[6],
)
PY
}

if ! run_evaluator; then
  show_log_tail
  if ! run_evaluator; then
    echo "retry failed, reverting candidate commit" >&2
    show_log_tail
    python - "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

from autoresearch.frontier import revert_last_commit

revert_last_commit(repo_root=Path(sys.argv[1]))
PY
    record_result "" "crash" "evaluation failed twice; see $LOG_PATH"
    echo "decision crash"
    exit 1
  fi
fi

SCORE="$(parse_score)"

if [[ -z "$BEST_BEFORE" ]] || python - "$SCORE" "$BEST_BEFORE" <<'PY'
import sys

candidate = float(sys.argv[1])
best = float(sys.argv[2])
raise SystemExit(0 if candidate > best else 1)
PY
then
  record_result "$SCORE" "keep" "kept with score $SCORE; log $LOG_PATH"
  echo "decision keep score=$SCORE sha=$CANDIDATE_SHA"
  exit 0
fi

python - "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

from autoresearch.frontier import revert_last_commit

revert_last_commit(repo_root=Path(sys.argv[1]))
PY

record_result "$SCORE" "discard" "discarded with score $SCORE; current best $BEST_BEFORE; log $LOG_PATH"
echo "decision discard score=$SCORE best=$BEST_BEFORE sha=$CANDIDATE_SHA"
