#!/bin/bash
# =====================================================================================================================
# Dynamic visualisation attribution tool and threat modelling: a local web page that runs the thesis tools on the
# evidence you type (3-D attribution) and builds possible attacks of a group (attack generator).
#
#   bash run_explorer.sh          six-feature tool (FinalToolsPhD_6feat), opens http://127.0.0.1:8765
#   bash run_explorer.sh 3f       three-feature tool (FinalToolsPhD_3feat)
#   Ctrl+C in this window stops it.
#
# Uses the tool's own Python (venv) and imports the tools read-only: no bytecode and no results file is written
# into FinalToolsPhD_6feat / _3feat.  names.json (display names from Cyber_-Necronomicon) is written next to this
# script the first time.
# =====================================================================================================================
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
exec "$PY" server.py --tool "$ROOT" --dumps "$PARENT/Cyber_-Necronomicon" "${@:2}"
