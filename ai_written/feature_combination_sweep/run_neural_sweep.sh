#!/bin/bash
# =====================================================================================================================
# Feature-combination sweep, the part that needs your Mac: XGBoost, GNN, Siamese and MIL must be RETRAINED for every
# feature configuration, and only FinalToolsPhD_6feat/venv has PyTorch and XGBoost.
#
# For each configuration a throw-away copy of FinalToolsPhD_6feat is made (code files byte-identical, every feature type
# outside the configuration set to -1 = missing in BaseData/EncodedData.csv), each method's own train.py is run inside
# the copy, and all eight methods are evaluated with the thesis protocol (sanity check, stress test, the 30 CTI reports;
# for the TTP-only configuration also the external evidence sets).  FinalToolsPhD_6feat and FinalToolsPhD_3feat are only
# read.  Everything new is written into this folder.
#
#   bash run_neural_sweep.sh             the TTP-only configuration (T), about 15-20 min.  The first time, your SAVED 6f
#                                        models are first run through the sweep code and must give exactly your saved
#                                        ranks: look at the lines starting with "VALIDATION" (each must read N/N identical)
#   bash run_neural_sweep.sh --check     only the checks, for the 6f AND 3f tools, plus the RF retrained on this Mac
#   CONFIGS=singles bash run_neural_sweep.sh   the six single feature types (a few hours)
#   CONFIGS=all bash run_neural_sweep.sh       all 63 configurations (very long, about 40 h; stop with Ctrl-C and start
#                                              again whenever you like: finished configurations are never redone)
#   CONFIGS=T,TS,TSV bash run_neural_sweep.sh  any list (letters: T techniques, S software, A CAPEC, W CWE, V CVE, P Pulse)
#   REPEATS=3 bash run_neural_sweep.sh   train the models 3 times per configuration (GNN, Siamese and MIL are trained
#                                        without a fixed seed, so this shows how much a retraining alone moves the results)
#
# The Mac is kept awake while it runs.  Log: logs/run_log.txt   (each training: logs/<configuration>/train_<method>.txt)
# =====================================================================================================================
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$(dirname "$HERE")")"          # .../claudePhDFolder
TOOL6="$PARENT/FinalToolsPhD_6feat"
TOOL3="$PARENT/FinalToolsPhD_3feat"
export MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1

if command -v caffeinate >/dev/null 2>&1 && [ -z "$PHD_CAFFEINATED" ]; then
    export PHD_CAFFEINATED=1
    exec caffeinate -i bash "$HERE/run_neural_sweep.sh" "$@"
fi

CHECK=""
for arg in "$@"; do
    case "$arg" in
        --check) CHECK=1 ;;
        *) echo "unknown option: $arg (use --check, or the CONFIGS / REPEATS variables)"; exit 1 ;;
    esac
done
CONFIGS="${CONFIGS:-T}"
REPEATS="${REPEATS:-1}"

mkdir -p "$HERE/logs"
exec > >(tee -a "$HERE/logs/run_log.txt") 2>&1
echo "=== neural sweep started $(date '+%Y-%m-%d %H:%M') ${CHECK:+(check only)} CONFIGS=$CONFIGS REPEATS=$REPEATS ==="

[ -d "$TOOL6" ] || { echo "ERROR: $TOOL6 not found"; exit 1; }
if [ -x "$TOOL6/venv/bin/python" ]; then PY="$TOOL6/venv/bin/python"
elif command -v python >/dev/null 2>&1; then PY="$(command -v python)"; else PY="$(command -v python3)"; fi
"$PY" -c "import sys, numpy, pandas, scipy, sklearn, matplotlib, seaborn, joblib, torch, torch_geometric, xgboost; print('python', sys.version.split()[0], 'from', sys.prefix)" || {
    echo "ERROR: $PY cannot import the required packages"; exit 1; }
TEMPLATE="$TOOL6/CTI_eval/CTI_Eval_empty_template.json"
EXTERNAL="$PARENT/FolderForNewEvaluationAttempt/inputs/external_sets.json"
ALL8="xgboost,gnn,siamese,mil,rf,jaccard,featureMatching,tvd"

# ---------------------------------------------------------------------------------------------------------- checks
check_6f() {
    echo; echo "--- check: your SAVED 6f models through the sweep code (nothing is retrained); every line must read N/N identical"
    "$PY" "$HERE/code/run_sweep.py" --tool-root "$TOOL6" --configs 6f --methods "$ALL8" --rf-saved \
        --out "$HERE/results_check/saved_models_6f" --validate --saved-6f "$TOOL6" 2>&1 | grep -v -i warn | sed 's/^\[validation\]/VALIDATION/' \
        || echo "  !! the check could not run completely (see above)"
}
if [ -n "$CHECK" ]; then
    check_6f
    if [ -d "$TOOL3" ]; then
        echo; echo "--- 3f: your saved models through the sweep code (nothing is retrained)"
        "$PY" "$HERE/code/run_sweep.py" --tool-root "$TOOL3" --configs 3f --methods "$ALL8" --rf-saved \
            --out "$HERE/results_check/saved_models_3f" --validate --saved-3f "$TOOL3" 2>&1 | grep -v -i warn | sed 's/^\[validation\]/VALIDATION/'
    fi
    echo; echo "--- RF retrained on this Mac for 6f and 3f, compared with your saved RF results"
    "$PY" "$HERE/code/run_sweep.py" --tool-root "$TOOL6" --configs 6f,3f --methods rf \
        --out "$HERE/results_check/rf_retrained" --validate --saved-6f "$TOOL6" --saved-3f "$TOOL3" 2>&1 | grep -v -i warn | sed 's/^\[validation\]/VALIDATION/'
    echo; echo "=== CHECK FINISHED $(date '+%Y-%m-%d %H:%M'): every VALIDATION line should read N/N identical ==="
    exit 0
fi

# ---------------------------------------------------------------------------------------------------------- sweep
# the 6f check runs once, before the first training (a few minutes)
[ -f "$HERE/results_check/saved_models_6f/validation.json" ] || check_6f
KEYS=$("$PY" -c "import sys; sys.path.insert(0, '$HERE/code'); from sweep_core import parse_configs; print(' '.join(parse_configs('$CONFIGS')))")
echo "configurations: $KEYS"
declare -a TRAIN_ORDER=("2.GNN" "3.SiameseNetwork" "1.XGboost" "4.Attention-Based_MIL")   # fastest first
model_file() {
    case "$1" in
        1.XGboost) echo "6feat_efficient_apt_similarity_nopca.pkl" ;;
        2.GNN) echo "6feat_trained_apt_gnn_model.pkl" ;;
        3.SiameseNetwork) echo "6feat_apt_siamese_feature_model.pkl" ;;
        4.Attention-Based_MIL) echo "6feat_apt_similarity_model_complete.pth" ;;
    esac
}
method_of() {
    case "$1" in
        1.XGboost) echo xgboost ;; 2.GNN) echo gnn ;; 3.SiameseNetwork) echo siamese ;; 4.Attention-Based_MIL) echo mil ;;
    esac
}

for KEY in $KEYS; do
  for REP in $(seq 1 "$REPEATS"); do
    TAG="$KEY"; OUT="$HERE/results_mac"
    if [ "$REP" -gt 1 ]; then TAG="${KEY}_repeat$REP"; OUT="$HERE/results_mac/repeat$REP"; fi
    mkdir -p "$OUT"
    if [ -f "$OUT/.done_$KEY" ]; then echo; echo "--- $TAG already done, skipped"; continue; fi
    echo; echo "=================== configuration $TAG  $(date '+%H:%M')"
    DEST="$HERE/_work/$TAG"
    [ -f "$DEST/CONFIG.json" ] || "$PY" "$HERE/code/build_config_tool.py" --tool-root "$TOOL6" --config "$KEY" --dest "$DEST"
    mkdir -p "$HERE/logs/$TAG"
    METHODS="rf,jaccard,featureMatching,tvd"
    for M in "${TRAIN_ORDER[@]}"; do
        MF="$DEST/$M/$(model_file "$M")"
        if [ ! -f "$MF" ]; then
            T0=$(date +%s)
            echo "  training $M ..."
            if (cd "$DEST/$M" && "$PY" train.py) > "$HERE/logs/$TAG/train_$M.txt" 2>&1; then
                echo "  trained $M in $(( $(date +%s) - T0 )) s"
            else
                echo "  !! TRAINING FAILED: $M (see logs/$TAG/train_$M.txt, last lines below)"; tail -n 15 "$HERE/logs/$TAG/train_$M.txt"
            fi
        fi
        [ -f "$MF" ] && METHODS="$(method_of "$M"),$METHODS"
    done
    EXTRA=""
    # The external evidence sets (inputs/external_sets.json) were built for the 181-entry dataset and are not used
    # in the thesis: they are no longer run with the TTP-only configuration (changed 23 November 2026).
    "$PY" "$HERE/code/run_sweep.py" --tool-root "$DEST" --configs "$KEY" --methods "$METHODS" --out "$OUT" \
        --cti-template "$TEMPLATE" $EXTRA 2>&1 | grep -v -i warn || echo "  !! run_sweep.py stopped with an error (see above)"
    N_OK=$(ls "$OUT"/ranks/*/"$KEY".npz 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_OK" -ge 8 ]; then
        touch "$OUT/.done_$KEY"
        # the models of the TTP-only tool are kept (about 1.2 GB) for later runs; the others are removed
        if [ "$KEY" != "T" ] || [ "$REP" -gt 1 ]; then
            for M in "${TRAIN_ORDER[@]}"; do rm -f "$DEST/$M/$(model_file "$M")"; done
        fi
    else
        echo "  !! only $N_OK of 8 methods finished for $TAG: the configuration will be retried next time"
    fi
  done
done

echo; echo "=== summary"
"$PY" "$HERE/code/analyse_sweep.py" --results "$HERE/results_mac" --quick 2>&1 | grep -v -i warn || true
echo; echo "=== FINISHED $(date '+%Y-%m-%d %H:%M') ==="
