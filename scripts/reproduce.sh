#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

backend="${MRI_BACKEND:-cpu}"
if [[ "$backend" != cpu && "$backend" != cuda ]]; then
  echo "MRI_BACKEND must be cpu or cuda" >&2
  exit 2
fi
uv sync --frozen --extra "$backend"
run="outputs/reproduce-$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --frozen --extra "$backend" python -m cvproj.train \
  "device=$backend" "hydra.run.dir=$run" "$@"

baseline="${MRI_BASELINE_DIR:-eval}"
if [[ ! -f "$baseline/baseline_metrics.json" || ! -f "$baseline/baseline_protocol.json" ]]; then
  echo "Training complete: $run. No reviewed baseline yet; regression gate not run." >&2
  echo "Review the full-data validation result before explicitly freezing a baseline." >&2
  exit 3
fi
uv run --frozen --extra "$backend" python scripts/eval_gate.py \
  --candidate "$run/metrics.json" --baseline "$baseline/baseline_metrics.json" \
  --rules eval/reconstruction_rules.json \
  --candidate-protocol "$run/protocol.json" --baseline-protocol "$baseline/baseline_protocol.json"
