#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-core}"
MODE="${2:---venv}"
PYTHON="${PYTHON:-python3}"
usage() { echo 'Usage: bash scripts/setup.sh [core|life|shopping|paperqa2|gaia] [--conda|--venv]' >&2; exit 2; }
case "$TARGET" in core|life|shopping|paperqa2|gaia) ;; *) usage;; esac
[[ $# -le 2 ]] || usage
case "$MODE" in
  --conda)
    if [[ -z "${CONDA_PREFIX:-}" || ! -d "$CONDA_PREFIX/conda-meta" || "${CONDA_DEFAULT_ENV:-}" == base ]]; then
      echo 'Create and activate a dedicated Conda environment first (see README).' >&2
      exit 2
    fi
    ENV_DIR="$CONDA_PREFIX"
    PYTHON="$ENV_DIR/bin/python"
    "$PYTHON" -c 'import os, pathlib, sys; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ["CONDA_PREFIX"]).resolve(), "Conda interpreter mismatch"'
    ;;
  --venv)
    ENV_DIR="${SHARP_VENV:-$ROOT/.venv-$TARGET}"
    ;;
  *) usage;;
esac
"$PYTHON" -c 'import sys; assert (3, 12) <= sys.version_info < (3, 14), "Use Python 3.12 or 3.13"'
if [[ "$MODE" == --venv && ! -x "$ENV_DIR/bin/python" ]]; then "$PYTHON" -m venv "$ENV_DIR"; fi
ENV_DIR="$(cd "$ENV_DIR" && pwd)"
"$ENV_DIR/bin/python" -m pip install -e "$ROOT"
if [[ "$TARGET" == core ]]; then exit 0; fi
"$ENV_DIR/bin/python" "$ROOT/scripts/fetch_upstream.py" "$TARGET"
case "$TARGET" in
  life) "$ENV_DIR/bin/python" -m pip install "$ROOT/external/life/TauBench" 'transformers>=4.57,<5' ;;
  shopping) "$ENV_DIR/bin/python" -m pip install -r "$ROOT/scripts/requirements-shopping.txt" ;;
  paperqa2)
    # Match the pinned upstream lockfile; LMI 1.x removed get_router().
    "$ENV_DIR/bin/python" -m pip install "$ROOT/external/paperqa2[local]" 'pypdf[image]' fonttools \
      'fhlmi==0.45.0' 'fhaviary==0.34.0'
    ;;
  gaia)
    "$ENV_DIR/bin/python" -m pip install uv
    UV_PROJECT_ENVIRONMENT="$ENV_DIR" "$ENV_DIR/bin/uv" sync --project "$ROOT/external/gaia" --frozen --no-dev --inexact
    ;;
esac
echo "Setup complete: $TARGET (environment: $ENV_DIR)"
