#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -v
OUT="$(mktemp -d "${TMPDIR:-/tmp}/sharp-smoke.XXXXXXXX")"
"$PYTHON" -m sharp.screen --input "$ROOT/examples/single_off.csv" \
  --components "$ROOT/examples/components.json" --output "$OUT/saliency.json"
echo "Synthetic smoke output: $OUT/saliency.json"
