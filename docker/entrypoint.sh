#!/usr/bin/env bash
set -euo pipefail

append_value_flag() {
  local flag="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    args+=("${flag}" "${value}")
  fi
}

append_bool_flag() {
  local flag="$1"
  local value="${2:-0}"
  case "${value,,}" in
    1|true|yes|on)
      args+=("${flag}")
      ;;
  esac
}

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

mode="${AUTORESEARCH_MODE:-demo}"

case "${mode}" in
  demo)
    args=(python -m autoresearch.demo)

    append_value_flag --config "${AUTORESEARCH_CONFIG:-}"
    append_value_flag --task "${AUTORESEARCH_TASK:-}"
    append_value_flag --iterations "${AUTORESEARCH_ITERATIONS:-}"
    append_value_flag --output-dir "${AUTORESEARCH_OUTPUT_DIR:-}"
    append_value_flag --agent-endpoint "${AUTORESEARCH_AGENT_ENDPOINT:-}"
    append_value_flag --agent-model "${AUTORESEARCH_AGENT_MODEL:-}"
    append_value_flag --prompt-preset "${AUTORESEARCH_PROMPT_PRESET:-}"
    append_value_flag --temperature "${AUTORESEARCH_TEMPERATURE:-}"
    append_bool_flag --trace "${AUTORESEARCH_TRACE:-0}"
    append_bool_flag --plot "${AUTORESEARCH_PLOT:-0}"

    exec "${args[@]}"
    ;;
  jupyter)
    port="${JUPYTER_PORT:-8888}"
    token="${JUPYTER_TOKEN:-autoresearch}"
    exec jupyter notebook \
      --ip=0.0.0.0 \
      --port="${port}" \
      --no-browser \
      --NotebookApp.token="${token}" \
      --NotebookApp.notebook_dir=/workspace \
      --NotebookApp.allow_remote_access=True
    ;;
  *)
    echo "Unsupported AUTORESEARCH_MODE: ${mode}" >&2
    exit 1
    ;;
esac
