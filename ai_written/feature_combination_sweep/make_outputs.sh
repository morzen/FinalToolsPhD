#!/bin/bash
# =====================================================================================================================
# Regenerate everything in tables/, figures/ and RESULTS.md from the saved rank matrices (no model is run, ~1 min):
#   results_cloud/   Jaccard, Feature Matching, TVD (and a cloud RF, replaced by results_mac_rf/) x the 63 combinations
#   results_mac_rf/  the RF x the 63 combinations, retrained with your venv
#   results_mac/     the Mac runs of run_neural_sweep.sh (all eight approaches per combination; repeatN/ = retrainings)
#   results_ttp_evidence/6f, 3f   run_ttp_evidence.sh: the thesis tools unchanged, queries with TTPs only
#   the SAVED results of FinalToolsPhD_6feat and FinalToolsPhD_3feat (read only)
#
#   bash make_outputs.sh
# =====================================================================================================================
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$(dirname "$HERE")")"
TOOL6="$PARENT/FinalToolsPhD_6feat"
TOOL3="$PARENT/FinalToolsPhD_3feat"
EXT="$PARENT/FolderForNewEvaluationAttempt"
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
if [ -x "$TOOL6/venv/bin/python" ]; then PY="$TOOL6/venv/bin/python"; else PY="$(command -v python3)"; fi
cd "$HERE/code"
# "sweep" = one result per (combination, approach), later folders replacing earlier ones: the cloud run (Jaccard,
# Feature Matching, TVD), the RF retrained with your venv (run_rf_sweep_mac.sh) and every combination run on the Mac
# (run_neural_sweep.sh: all eight approaches, first training).  "mac" / "mac_rN" = the Mac runs as they are.
RES=""                                               # results_cloud/ is optional (the Mac runs hold all eight approaches)
[ -d "$HERE/results_cloud/ranks" ] && RES="--results sweep=$HERE/results_cloud"
[ -d "$HERE/results_mac_rf/ranks" ] && RES="$RES --results sweep=$HERE/results_mac_rf"
[ -d "$HERE/results_mac/ranks" ] && RES="$RES --results sweep=$HERE/results_mac --results mac=$HERE/results_mac"
for R in "$HERE"/results_mac/repeat*; do            # retrainings of the same configurations (REPEATS=n)
    [ -d "$R/ranks" ] && RES="$RES --results mac_r${R##*repeat}=$R"
done
# TTP-only evidence with the thesis tools unchanged (run_ttp_evidence.sh): "ttpev6f" / "ttpev3f"
for T in 6f 3f; do
    [ -d "$HERE/results_ttp_evidence/$T/ranks" ] && RES="$RES --results ttpev$T=$HERE/results_ttp_evidence/$T"
done
"$PY" analyse_sweep.py --tool-root "$TOOL6" $RES --thesis "thesis6f=$TOOL6" --thesis "thesis3f=$TOOL3" \
    --out "$HERE/tables/sweep_metrics.csv"
"$PY" make_tables.py --metrics "$HERE/tables/sweep_metrics.csv" --out "$HERE/tables"
if [ -d "$HERE/results_mac/ranks_external" ]; then
    "$PY" external_ttp.py --results "$HERE/results_mac" --sets "$EXT/inputs/external_sets.json" \
        --summary "$EXT/tables/analysis_summary.json" --tool-root "$TOOL6" --out "$HERE/tables"
fi
"$PY" make_graphs.py --metrics "$HERE/tables/sweep_metrics.csv" --out "$HERE/figures"
"$PY" make_results_md.py --root "$HERE"
echo "done: tables/, figures/, RESULTS.md"
