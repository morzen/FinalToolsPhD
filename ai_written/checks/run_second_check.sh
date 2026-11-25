#!/bin/bash
# =====================================================================================================================
# SECOND CHECK OF APPENDIX F: every number of the result files recomputed from the saved rankings.
#
#     bash ai_written/checks/run_second_check.sh            about 1 minute
#
# second_check/recheck_numbers_163.py uses numpy and pandas only and none of the code of the tools. It reads saved
# files only and changes no result. It needs the rank matrices that run_all.sh writes in each results/ folder.
# Each run replaces second_check/second_check_output_6feat.txt and second_check_output_3feat.txt.
# =====================================================================================================================
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PHD_ROOT:-$(cd "$HERE/../.." && pwd)}"
if [ -n "$PHD_PY" ]; then PY="$PHD_PY"
else
    PY="$ROOT/FinalToolsPhD_6feat/venv/bin/python"
    "$PY" -c "import numpy, pandas" >/dev/null 2>&1 || PY="$(command -v python3)"
fi
export PYTHONDONTWRITEBYTECODE=1
ALL_OK=1
cd "$ROOT" || exit 1
for V in 6feat 3feat; do
    OUT="$HERE/second_check/second_check_output_$V.txt"
    {   echo "=== second check of Appendix F, FinalToolsPhD_$V, run on $(date '+%Y-%m-%d %H:%M %Z')"
        "$PY" -B "$HERE/second_check/recheck_numbers_163.py" "FinalToolsPhD_$V" || echo "  !! the script stopped with an error"
    } 2>&1 | tee "$OUT"
    grep -q "RESULT: ALL CONSISTENT" "$OUT" || ALL_OK=0
    echo
done
echo "====================================================================================================="
if [ "$ALL_OK" = 1 ]; then echo "=== BOTH TOOLS: ALL CONSISTENT. The two outputs are in second_check/."
else echo "=== A PROBLEM WAS REPORTED: see the lines marked PROBLEM or !! above."; fi
echo "====================================================================================================="
