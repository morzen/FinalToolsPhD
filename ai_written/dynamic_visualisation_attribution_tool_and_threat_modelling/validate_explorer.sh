#!/bin/bash
# Checks that the explorer shows exactly what the thesis tools compute (about 10-15 minutes; nothing is written).
#   bash validate_explorer.sh        six-feature tool
#   bash validate_explorer.sh 3f     three-feature tool
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PARENT="${PHD_DIR:-$(dirname "$(dirname "$HERE")")}"     # the folder holding FinalToolsPhD_6feat and Cyber_-Necronomicon
case "${1:-6f}" in
    3f) ROOT="$PARENT/FinalToolsPhD_3feat" ;;
    *)  ROOT="$PARENT/FinalToolsPhD_6feat" ;;
esac
PY="$ROOT/venv/bin/python"
[ -x "$PY" ] || PY="$PARENT/FinalToolsPhD_6feat/venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
cd "$HERE"
exec "$PY" validate_explorer.py --tool "$ROOT" "${@:2}"
