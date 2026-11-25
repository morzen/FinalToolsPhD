#!/bin/bash
# =====================================================================================
# Re-run every evaluation of this folder with the shared, thesis-conformant protocol
# (common/protocol.py). From this folder (no need to activate anything: the folder's own
# venv/bin/python is used automatically):
#
#   bash run_all.sh --smoke      quick end-to-end check of every script (~10-20 min)
#   bash run_all.sh              full run: XGBoost retraining, the 8 attribution evaluations,
#                                the 8 false-flag evaluations, the hybrid (fusion), the CTI
#                                evaluation, then collect_results.py
#   bash run_all.sh --resume     continue a full run that stopped: steps already completed
#                                are skipped (also works with --smoke)
#   bash run_all.sh --hybrid     only the hybrid (fusion) steps: hybrid_eval.py, hybrid_false_flag.py,
#                                CTI_eval.py, then collect_results.py (~2-3 min). Use it after changing
#                                HYBRID_WEIGHTS in common/protocol.py: the 8 methods are not re-run,
#                                their saved results are fused again with the new weights.
#                                Log: run_all_log_hybrid.txt, times: run_times_hybrid.txt
#   bash run_all.sh --heatmaps   only the original heatmap code, drawn from the saved final results
#                                (no evaluation is re-run, no result file is changed, ~5-10 min):
#                                the similarity heatmaps of Jaccard / Feature Matching / TVD, the false-
#                                flag heatmap of each method, and the hybrid heatmaps (one per weight
#                                configuration, in 6.hybrid/results/<configuration>/). The similarity
#                                heatmaps of the 5 ML models are drawn by their eval.py in a full run.
#                                Log: run_all_log_heatmaps.txt, times: run_times_heatmaps.txt
#
# Results: each method's results/ folder (overwritten by the next run), and a permanent copy
# of everything in results_archive/<date_time>/ (summary.txt, excel_blocks.csv, all files).
# The whole screen output is saved in run_all_log.txt, the time of every step in run_times.txt.
# =====================================================================================
set -eo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE

SMOKE=""; RESUME=""; HYBRID=""; HEATMAPS=""
for arg in "$@"; do
    case "$arg" in
        --smoke)  SMOKE="--smoke" ;;
        --resume) RESUME=1 ;;
        --hybrid) HYBRID=1 ;;
        --heatmaps) HEATMAPS=1 ;;
        *) echo "unknown option: $arg (use --smoke, --resume, --hybrid or --heatmaps)"; exit 1 ;;
    esac
done
SUFFIX=${SMOKE:+_smoke}                  # output file of each step
RUNSUFFIX=$SUFFIX${HYBRID:+_hybrid}${HEATMAPS:+_heatmaps}      # log / times / progress file of this run

# Keep the Mac awake while this runs (it would otherwise pause when the Mac goes to sleep)
SCRIPT="$PWD/$(basename "$0")"
if command -v caffeinate >/dev/null 2>&1 && [ -z "$PHD_CAFFEINATED" ]; then
    export PHD_CAFFEINATED=1
    exec caffeinate -i bash "$SCRIPT" "$@"
fi

# Python: this folder's venv (works even though the venv was created in another folder)
if [ -x "$PWD/venv/bin/python" ]; then PY="$PWD/venv/bin/python"
elif command -v python >/dev/null 2>&1; then PY="$(command -v python)"
else PY="$(command -v python3)"; fi

LOG="$PWD/run_all_log$RUNSUFFIX.txt"
TIMES="$PWD/run_times$RUNSUFFIX.txt"
DONE="$PWD/.run_all_done$RUNSUFFIX"
if [ -z "$RESUME" ]; then rm -f "$DONE"; : > "$LOG"; : > "$TIMES"; fi
exec > >(tee -a "$LOG") 2>&1

echo "=== run_all.sh ${SMOKE:-full run}${HYBRID:+ (hybrid steps only)}${HEATMAPS:+ (original heatmaps only)}${RESUME:+ (resume)} started $(date '+%Y-%m-%d %H:%M') ==="
"$PY" -c "import sys, numpy, pandas, scipy, sklearn, matplotlib, seaborn, joblib, torch, torch_geometric, xgboost; print('Python', sys.version.split()[0], 'from', sys.prefix)" || {
    echo "ERROR: $PY cannot import the required packages (see the message above)."; exit 1; }

# run_step <name> <command...>: runs once, records its duration; skipped with --resume if already done
run_step() {
    local name="$1"; shift
    if [ -n "$RESUME" ] && grep -qxF "$name" "$DONE" 2>/dev/null; then
        echo; echo "--- skipping $name (already completed)"; return 0
    fi
    echo; echo "=================== $name $SMOKE ==================="
    local t0; t0=$(date +%s)
    "$@"
    local dt=$(( $(date +%s) - t0 ))
    echo "$name" >> "$DONE"
    printf "%-55s %4d min %02d s\n" "$name" $((dt / 60)) $((dt % 60)) | tee -a "$TIMES"
}
# in_dir <folder> <log name> <script> [args]: runs a script inside its folder, log in results/
in_dir() {
    local dir="$1" log="$2"; shift 2
    mkdir -p "$dir/results"
    (cd "$dir" && "$PY" "$@" $SMOKE 2>&1 | tee "results/$log$SUFFIX.txt")
}

run_step "false-flag evidence" "$PY" -m common.protocol describe     # builds BaseData/false_flag_evidence.pkl if missing
if [ -n "$HEATMAPS" ]; then
    # The original heatmap code only, from the saved final results (--heatmaps-only: nothing is re-evaluated)
    for e in Jaccards_Correlation:Jaccards_correlation.py featureMatching:featureMatching.py TVD_Total_Variation_Distance:TVD_featureIng.py; do
        run_step "${e%%:*}/${e##*:} --heatmaps-only" in_dir "${e%%:*}" heatmap_similarity_output "${e##*:}" --heatmaps-only
    done
    for f in Jaccards_Correlation:false_flag.py featureMatching:false_flag.py TVD_Total_Variation_Distance:false_flag.py 1.XGboost:False_flag.py 2.GNN:False_Flag.py 3.SiameseNetwork:False_flag.py 4.Attention-Based_MIL:False_Flag.py 5.RF_FeatureEngineering:False_flag.py; do
        run_step "${f%%:*}/${f##*:} --heatmaps-only" in_dir "${f%%:*}" heatmap_false_flag_output "${f##*:}" --heatmaps-only
    done
    run_step "6.hybrid/hybrid_eval.py --heatmaps-only"       in_dir 6.hybrid heatmap_hybrid_eval_output hybrid_eval.py --heatmaps-only
    run_step "6.hybrid/hybrid_false_flag.py --heatmaps-only" in_dir 6.hybrid heatmap_hybrid_false_flag_output hybrid_false_flag.py --heatmaps-only
    echo; echo "=== ORIGINAL HEATMAPS DONE $(date '+%Y-%m-%d %H:%M') ==="; echo; cat "$TIMES"
    exit 0
fi
if [ -z "$HYBRID" ]; then
    # GNN, Siamese, MIL and Random Forest are loaded from their saved files: this script does not train them.
    # If one is missing (for instance after the dataset changed), train it with the train.py of its folder.
    for f in 2.GNN/*_trained_apt_gnn_model.pkl 3.SiameseNetwork/*_apt_siamese_feature_model.pkl \
             4.Attention-Based_MIL/*_apt_similarity_model_complete.pth 5.RF_FeatureEngineering/*_apt_similarity_model.joblib; do
        if [ ! -s "$f" ]; then
            echo; echo "STOPPED: the saved model $f is missing."
            echo "Train it with the train.py of its folder (python train.py, run inside that folder),"
            echo "then start this script again with --resume."
            exit 1
        fi
    done
    if [ -z "$SMOKE" ]; then
        # XGBoost is retrained so that -1 (missing value) is no longer a feature. It is trained BEFORE check_uniform,
        # which loads every model and needs each of them to match the dataset.
        run_step "1.XGboost/train.py" bash -c "set -o pipefail; mkdir -p 1.XGboost/results && cd 1.XGboost && \"$PY\" train.py 2>&1 | tee results/train_output.txt"
    fi
    run_step "check_uniform" bash -c "set -o pipefail; \"$PY\" check_uniform.py 2>&1 | tee check_uniform_output.txt"
fi

EVALS="Jaccards_Correlation:Jaccards_correlation.py featureMatching:featureMatching.py TVD_Total_Variation_Distance:TVD_featureIng.py 1.XGboost:eval.py 2.GNN:eval.py 3.SiameseNetwork:eval.py 4.Attention-Based_MIL:eval.py 5.RF_FeatureEngineering:eval.py"
FFS="Jaccards_Correlation:false_flag.py featureMatching:false_flag.py TVD_Total_Variation_Distance:false_flag.py 1.XGboost:False_flag.py 2.GNN:False_Flag.py 3.SiameseNetwork:False_flag.py 4.Attention-Based_MIL:False_Flag.py 5.RF_FeatureEngineering:False_flag.py"

if [ -z "$HYBRID" ]; then
    for e in $EVALS; do run_step "${e%%:*}/${e##*:}" in_dir "${e%%:*}" eval_output "${e##*:}"; done
    for f in $FFS;   do run_step "${f%%:*}/${f##*:}" in_dir "${f%%:*}" false_flag_output "${f##*:}"; done
fi
run_step "6.hybrid/hybrid_eval.py"        in_dir 6.hybrid hybrid_eval_output hybrid_eval.py
run_step "6.hybrid/hybrid_false_flag.py"  in_dir 6.hybrid hybrid_false_flag_output hybrid_false_flag.py
if [ -z "$SMOKE" ]; then
    run_step "CTI_eval/CTI_eval.py" bash -c "set -o pipefail; cd CTI_eval && \"$PY\" CTI_eval.py 2>&1 | tee CTI_eval_output.txt"
fi

echo; echo "=== ALL STEPS COMPLETED${SMOKE:+ (smoke mode)} $(date '+%Y-%m-%d %H:%M') ==="
echo; cat "$TIMES"
echo
"$PY" collect_results.py $SMOKE
