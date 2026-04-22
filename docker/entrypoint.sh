#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

mode="${AUTORESEARCH_MODE:-eval}"
experiment="${AUTORESEARCH_EXPERIMENT:-autoresearch/experiments/restaurant_train.py}"

case "${mode}" in
  eval)
    exec python -m autoresearch.experiments.restaurant_eval --experiment "${experiment}"
    ;;
  shell)
    exec bash
    ;;
  *)
    echo "Unsupported AUTORESEARCH_MODE: ${mode}" >&2
    exit 1
    ;;
esac
