#!/bin/bash
# =====================================================================================================================
# The Random Forest for all 63 feature combinations with YOUR venv, so that the RF points of the sweep figure use the
# same scikit-learn as the thesis (Tables 4.5 / 4.6).  The cloud's scikit-learn breaks ties between equally good splits
# differently (6f: 7 of 100 trees, one query; other combinations: up to 8 queries).  The RF is retrained in memory with
# the classes of 5.RF_FeatureEngineering/train.py: nothing is written in FinalToolsPhD_6feat.  About 15 min (run on
# 2026-10-29: 6f and 3f identical to your saved RF); the Mac stays awake.
#
#   bash run_rf_sweep_mac.sh          results in results_mac_rf/ ; the last lines compare the 6f and 3f combinations with
#                                     your saved RF results (VALIDATION, should read N/N identical)
# Then:  bash make_outputs.sh         (the sweep figures and tables take the RF from results_mac_rf/)
# =====================================================================================================================
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$(dirname "$HERE")")"
TOOL6="$PARENT/FinalToolsPhD_6feat"
TOOL3="$PARENT/FinalToolsPhD_3feat"
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1
if command -v caffeinate >/dev/null 2>&1 && [ -z "$PHD_CAFFEINATED" ]; then
    export PHD_CAFFEINATED=1
    exec caffeinate -i bash "$HERE/run_rf_sweep_mac.sh" "$@"
fi
if [ -x "$TOOL6/venv/bin/python" ]; then PY="$TOOL6/venv/bin/python"; else PY="$(command -v python3)"; fi
mkdir -p "$HERE/logs"
exec > >(tee -a "$HERE/logs/run_log_rf.txt") 2>&1
echo "=== RF sweep started $(date '+%Y-%m-%d %H:%M') ==="
"$PY" "$HERE/code/run_sweep.py" --tool-root "$TOOL6" --configs all --methods rf --out "$HERE/results_mac_rf" \
    --validate --saved-6f "$TOOL6" --saved-3f "$TOOL3" 2>&1 | grep -v -i warn | sed 's/^\[validation\]/VALIDATION/'
echo "=== FINISHED $(date '+%Y-%m-%d %H:%M') ==="
