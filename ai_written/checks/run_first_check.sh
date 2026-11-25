#!/bin/bash
# =====================================================================================================================
# FIRST CHECK OF APPENDIX F: the final code against the original scoring code, on the same queries.
#
#     bash ai_written/checks/run_first_check.sh             about 40 minutes for the two tools
#
# It needs the saved models of the two tool folders and their Python environment, and changes no result.
# Options are passed on to verify_against_original.py, for instance  --methods jaccard,tvd  --pairs 2 --apts 2
# =====================================================================================================================
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PHD_ROOT:-$(cd "$HERE/../.." && pwd)}"
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1
FAILED=""
for V in 6feat 3feat; do
    T="$ROOT/FinalToolsPhD_$V"
    if [ -n "$PHD_PY" ]; then PY="$PHD_PY"
    elif [ -x "$T/venv/bin/python" ]; then PY="$T/venv/bin/python"
    elif [ -x "$ROOT/FinalToolsPhD_6feat/venv/bin/python" ]; then PY="$ROOT/FinalToolsPhD_6feat/venv/bin/python"
    else PY="$(command -v python3)"; fi
    echo; echo "=================== first check, FinalToolsPhD_$V   $(date '+%H:%M')"
    (cd "$T" && PHD_TOOL_DIR="$T" "$PY" -B "$HERE/first_check/verify_against_original.py" \
        --backup "$HERE/first_check/original_code/FinalToolsPhD_$V" "$@") || FAILED="$FAILED $V"
    [ -f "$T/verify_report.txt" ] && mv "$T/verify_report.txt" "$HERE/first_check/verify_report_$V.txt"
done
echo
echo "====================================================================================================="
if [ -z "$FAILED" ]; then echo "=== FIRST CHECK FINISHED: the reports are first_check/verify_report_6feat.txt and _3feat.txt"
else echo "=== FIRST CHECK FINISHED WITH DIFFERENCES OR ERRORS IN:$FAILED"; fi
echo "====================================================================================================="
