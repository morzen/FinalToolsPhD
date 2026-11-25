#!/bin/bash
# =====================================================================================================================
# TTP-only EVIDENCE with your two thesis tools exactly as they are: the six-feature tool and the three-feature tool,
# their profiles and their SAVED, trained models (nothing is retrained), and the same three tests as Tables 4.5 and
# 4.6 (sanity check, stress test - shuffled and divided, the 30 CTI reports).  The only change: every query keeps only
# its TTPs.  The TTP half of each stress-test query is the one of Tables 4.5 / 4.6 (same seeded split).
# Nothing in FinalToolsPhD_6feat or FinalToolsPhD_3feat is written.  About 10-20 minutes; it can run in a second
# Terminal window while run_neural_sweep.sh is running.
#
#   bash run_ttp_evidence.sh      results: results_ttp_evidence/6f and results_ttp_evidence/3f
#                                 log:     logs/run_log_ttp_evidence.txt
#
# First, once: your saved 3f models are run through this code with ALL their evidence and compared with your saved
# results (the eight lines under "VALIDATION" must read 181/181 rows identical and 30/30 reports identical, i.e. Table
# 4.6 is reproduced).  The same check for the 6f tool was done on 28 October (results_check/saved_models_6f: 8 of 8
# identical).
# =====================================================================================================================
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$(dirname "$HERE")")"
TOOL6="$PARENT/FinalToolsPhD_6feat"
TOOL3="$PARENT/FinalToolsPhD_3feat"
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1
if command -v caffeinate >/dev/null 2>&1 && [ -z "$PHD_CAFFEINATED_EV" ]; then
    export PHD_CAFFEINATED_EV=1
    exec caffeinate -i bash "$HERE/run_ttp_evidence.sh" "$@"
fi
PY6="$TOOL6/venv/bin/python"; [ -x "$PY6" ] || PY6="$(command -v python3)"
PY3="$TOOL3/venv/bin/python"; [ -x "$PY3" ] || PY3="$PY6"           # each tool with its own venv
TEMPLATE="$TOOL6/CTI_eval/CTI_Eval_empty_template.json"              # the same file in both tools
ALL8="xgboost,gnn,siamese,mil,rf,jaccard,featureMatching,tvd"
mkdir -p "$HERE/logs"
exec > >(tee -a "$HERE/logs/run_log_ttp_evidence.txt") 2>&1
echo "=== TTP-only evidence started $(date '+%Y-%m-%d %H:%M') ==="
cd "$HERE"

if [ ! -f "$HERE/results_check/saved_models_3f/validation.json" ]; then
    echo; echo "--- check: your SAVED 3f models through this code, all evidence (the lines under VALIDATION must read 181/181 and 30/30)"
    "$PY3" code/run_sweep.py --tool-root "$TOOL3" --configs 3f --methods "$ALL8" --rf-saved --cti-template "$TEMPLATE" \
        --out "$HERE/results_check/saved_models_3f" --validate --saved-3f "$TOOL3" 2>&1 | grep -v -i warn \
        | sed 's/^\[validation\]/VALIDATION/' || echo "  !! the check could not run completely (see above)"
fi

echo; echo "--- six-feature tool, TTP-only evidence"
"$PY6" code/run_evidence_only.py --tool-root "$TOOL6" --evidence T --cti-template "$TEMPLATE" \
    --out "$HERE/results_ttp_evidence/6f" 2>&1 | grep -v -i warn || echo "  !! stopped with an error (see above)"

echo; echo "--- three-feature tool, TTP-only evidence"
"$PY3" code/run_evidence_only.py --tool-root "$TOOL3" --evidence T --cti-template "$TEMPLATE" \
    --out "$HERE/results_ttp_evidence/3f" 2>&1 | grep -v -i warn || echo "  !! stopped with an error (see above)"

echo; echo "=== FINISHED $(date '+%Y-%m-%d %H:%M') ==="
