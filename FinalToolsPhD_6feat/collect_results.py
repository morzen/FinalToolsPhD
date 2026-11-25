"""
Collect every result of this folder into one place, after run_all.sh (it is called at the end).

Writes results_archive/<date_time>/ (results_archive/smoke_<date_time>/ with --smoke) containing
  * summary.txt        all results as readable tables (percentages);
  * excel_blocks.csv   the same numbers as fractions (0-1), laid out block by block like
                       Fin_result.xlsx (same method order, same columns), with the target cells;
  * a copy of every result file of the run (JSON summaries, logs, heatmaps; not the large .npy
    matrices), so a later re-run never overwrites the record of this one.

Run from this folder:  python collect_results.py   [--smoke]
"""
import os
import sys
import csv
import json
import glob
import shutil
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import protocol as P

SMOKE = "--smoke" in sys.argv
SIX = len(P.FEATURES) == 6
VERSION = "6 FEATURES" if SIX else "3 FEATURES"

# Column order of Fin_result.xlsx
ATTR_COLS = [('xgboost', 'Xgboost'), ('gnn', 'GNN'), ('siamese', 'Siamese'), ('mil', 'MIL'), ('rf', 'RF'),
             ('featureMatching', 'FeatureMatch'), ('jaccard', "Jaccard's"), ('tvd', 'TVD'),
             ('hybrid:no_weights', 'Hybrid no weight'), ('hybrid:self_eval_weights', 'weighted_self_eval'),
             ('hybrid:CTI_boosting', 'weighted_Cti_boosting')]
FF_COLS = [('xgboost', 'Xgboost'), ('gnn', 'GNN'), ('siamese', 'Siamese'), ('mil', 'MIL'), ('rf', 'RF'),
           ('featureMatching', 'Feature match'), ('jaccard', 'jaccards'), ('tvd', 'TVD'),
           ('hybrid:no_weights', 'hybrid'), None, ('hybrid:self_eval_weights', 'hybrid_self_eval_weights'),
           None, ('hybrid:CTI_boosting', 'hybrid_CTI_boosting (not in the sheet yet)')]
CTI_KEY = {'xgboost': 'XGboost', 'gnn': 'GNN', 'siamese': 'Siamesenetwork', 'mil': 'MIL', 'rf': 'RF',
           'jaccard': 'Jacccards', 'tvd': 'TVD', 'featureMatching': 'featureMatching',
           'hybrid:no_weights': 'hybrid_no_weights', 'hybrid:self_eval_weights': 'hybrid_self_eval_weights',
           'hybrid:CTI_boosting': 'hybrid_CTI_boosting'}
# Target cells in Fin_result.xlsx (6-feature block at the top, 3-feature block below)
CELLS = {True: {'attr_full': 'D7:N10', 'attr_divided': 'D12:N15', 'cti_rank': 'D17:N17', 'cti_score': 'D18:N18',
                'ff_full': 'T6:BH9', 'ff_partial': 'T11:BH14'},
         False: {'attr_full': 'D46:N49', 'attr_divided': 'D51:N54', 'cti_rank': 'D56:N56', 'cti_score': 'D57:N57',
                 'ff_full': 'T50:BH53', 'ff_partial': 'T55:BH58'}}[SIX]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def results_dir(folder):
    d = os.path.join(HERE, folder, "results")
    return os.path.join(d, "smoke") if SMOKE else d


def same_weights(rec):
    """True if a hybrid result file records exactly the weights now in common/protocol.py"""
    try:
        return all(abs(float(rec[c][m]) - w[m]) < 1e-12 for c, w in P.HYBRID_WEIGHTS.items() for m in w)
    except (KeyError, TypeError, ValueError):
        return False


def gather():
    attr, ff, missing = {}, {}, []
    fp = P.evidence_fingerprint() if os.path.exists(P.FF_EVIDENCE_FILE) else None
    for m in P.METHODS:
        d = results_dir(P.METHOD_DIRS[m])
        a = load(os.path.join(d, "attribution_eval.json"))
        f = load(os.path.join(d, "false_flag_summary.json"))
        if a:
            attr[m] = {s: {int(k): v / 100.0 for k, v in a['results'][s]['top_k'].items()} for s in ("full", "divided")}
        else:
            missing.append(f"{P.METHOD_DIRS[m]}/results/attribution_eval.json")
        if f and f.get('evidence_fingerprint') == fp:
            ff[m] = {s: {int(k): {x: v / 100.0 for x, v in r.items()} for k, r in f['results'][s].items()}
                     for s in ("full", "partial")}
            ff.setdefault('_meta', f)
        elif f:
            missing.append(f"{P.METHOD_DIRS[m]}/results/false_flag_summary.json is STALE "
                           f"(computed on other false-flag evidence): not used")
        else:
            missing.append(f"{P.METHOD_DIRS[m]}/results/false_flag_summary.json")
    d = results_dir("6.hybrid")
    ha = load(os.path.join(d, "hybrid_attribution_eval.json"))
    hf = load(os.path.join(d, "hybrid_false_flag_summary.json"))
    rerun = "computed with other hybrid weights than common/protocol.py (re-run: bash run_all.sh --hybrid): not used"
    if ha and not same_weights(ha.get('_weights')):
        missing.append(f"6.hybrid/results/hybrid_attribution_eval.json {rerun}"); ha = None
    if hf and not same_weights(hf.get('weights')):
        missing.append(f"6.hybrid/results/hybrid_false_flag_summary.json {rerun}"); hf = {'evidence_fingerprint': fp, 'results': {}}
    for cfg in P.HYBRID_WEIGHTS:
        if ha and cfg in ha:
            attr[f"hybrid:{cfg}"] = {s: {int(k): v / 100.0 for k, v in ha[cfg][s].items()} for s in ("full", "divided")}
        if hf and hf.get('evidence_fingerprint') == fp and cfg in hf['results']:
            ff[f"hybrid:{cfg}"] = {s: {int(k): {x: v / 100.0 for x, v in r.items()} for k, r in hf['results'][cfg][s].items()}
                                   for s in ("full", "partial")}
    if not ha and not any(m.startswith("6.hybrid/results/hybrid_attribution_eval.json") for m in missing):
        missing.append("6.hybrid/results/hybrid_attribution_eval.json")
    if not hf:
        missing.append("6.hybrid/results/hybrid_false_flag_summary.json")
    elif hf.get('evidence_fingerprint') != fp:
        missing.append("6.hybrid/results/hybrid_false_flag_summary.json is STALE "
                       "(computed on other false-flag evidence): not used")
    cti = None if SMOKE else load(os.path.join(HERE, "CTI_eval", "CTI_summary.json"))
    if not SMOKE and not cti:
        missing.append("CTI_eval/CTI_summary.json")
    if cti and not same_weights(cti.get('hybrid_weights')):
        missing.append(f"CTI_eval/CTI_summary.json: hybrid columns {rerun}")
        for part in ('cti_score', 'mean_rank'):
            cti[part] = {c: v for c, v in cti[part].items() if not c.startswith("hybrid_")}
    return attr, ff, cti, missing


def fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{100 * v:.2f}" if pct else f"{v:.2f}"


def text_report(attr, ff, cti, missing):
    L = [f"{VERSION} VERSION - results collected {datetime.datetime.now():%Y-%m-%d %H:%M}"
         + ("  (SMOKE RUN: tiny subsets, not thesis results)" if SMOKE else "")]
    meta = ff.get('_meta')
    if meta:
        L.append(f"False-flag evidence: {meta['n_pairs']} ordered pairs, {meta['n_excluded']} excluded, "
                 f"rule: {meta['params'].get('rule', '?')} (fingerprint {meta['evidence_fingerprint'][:12]})")
    L += ["", "Hybrid weights (common/protocol.py, HYBRID_WEIGHTS):"] + P.hybrid_weights_lines()
    if missing:
        L += ["", "MISSING (step not run or failed):"] + [f"  {m}" for m in missing]
    for s, title in (("full", "ATTRIBUTION - FULL DATA"), ("divided", "ATTRIBUTION - SHUFFLED & DIVIDED")):
        L += ["", f"{title}  (Top-k precision, %)", f"{'method':26s}" + "".join(f"{'Top ' + str(k):>9s}" for k in P.TOP_K)]
        for key, name in ATTR_COLS:
            row = attr.get(key, {}).get(s, {})
            L.append(f"{name:26s}" + "".join(f"{fmt(row.get(k)):>9s}" for k in P.TOP_K))
    for s, title in (("full", "FALSE FLAG - FULL"), ("partial", "FALSE FLAG - PARTIAL")):
        L += ["", f"{title}  (% of pairs: AD attacker detected, MA mimicked APT, Both, N neither)",
              f"{'method':26s}" + "".join(f"{x + '@' + str(k):>9s}" for k in P.TOP_K for x in ("AD", "MA", "Both", "N"))]
        for col in FF_COLS:
            if col is None:
                continue
            key, name = col
            row = ff.get(key, {}).get(s, {})
            vals = [row.get(k, {}).get(x) for k in P.TOP_K for x in ("AD", "MA", "Both", "Neither")]
            L.append(f"{name.split(' (')[0]:26s}" + "".join(f"{fmt(v):>9s}" for v in vals))
    if cti:
        L += ["", f"CTI EVALUATION  ({cti['n_reports']} reports; score = 100 (1 - (rank - 1) / (N - 1)))",
              f"{'method':26s}{'CTI score %':>12s}{'mean rank':>11s}"]
        for key, name in ATTR_COLS:
            c = CTI_KEY[key]
            L.append(f"{name:26s}{fmt(cti['cti_score'].get(c), pct=False):>12s}{fmt(cti['mean_rank'].get(c), pct=False):>11s}")
    return "\n".join(L) + "\n"


def excel_blocks(attr, ff, cti, path):
    def num(v):
        return "" if v is None else f"{v:.4f}"

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# {VERSION} VERSION. Values are fractions (0-1) like Fin_result.xlsx. In each block the first "
                    f"column is only a label; the values go into the cells named in the block title."])
        for s, cells, title in (("full", CELLS['attr_full'], "Attribution - Full Data"),
                                ("divided", CELLS['attr_divided'], "Attribution - Shuffled N Divided")):
            w.writerow([])
            w.writerow([f"# {title} -> {cells}"])
            w.writerow(["top"] + [n for _, n in ATTR_COLS])
            for k in P.TOP_K:
                w.writerow([k] + [num(attr.get(key, {}).get(s, {}).get(k)) for key, _ in ATTR_COLS])
        if cti:
            w.writerow([])
            w.writerow([f"# CTI average rank ({cti['n_reports']} reports) -> {CELLS['cti_rank']}"])
            w.writerow([""] + [n for _, n in ATTR_COLS])
            w.writerow(["mean rank"] + [f"{cti['mean_rank'].get(CTI_KEY[k], float('nan')):.2f}" for k, _ in ATTR_COLS])
            w.writerow([f"# CTI score, thesis formula 100(1-(rank-1)/(N-1)) as a fraction -> {CELLS['cti_score']} "
                        f"(replaces the old '1 - rank/181' formula)"])
            w.writerow(["CTI score"] + [num(cti['cti_score'].get(CTI_KEY[k], float('nan')) / 100.0) for k, _ in ATTR_COLS])
        for s, cells, title in (("full", CELLS['ff_full'], "False flag - Full Data"),
                                ("partial", CELLS['ff_partial'], "False flag - shuffle and divided (partial)")):
            w.writerow([])
            w.writerow([f"# {title} -> {cells}; the hybrid_CTI_boosting columns after it are extra (not in the sheet yet)"])
            header, sub = ["top"], [""]
            for col in FF_COLS:
                if col is None:
                    header.append("")
                    sub.append("")
                    continue
                header += [col[1], "", "", ""]
                sub += ["AD", "MA", "Both", "N"]
            w.writerow(header)
            w.writerow(sub)
            for k in P.TOP_K:
                row = [k]
                for col in FF_COLS:
                    if col is None:
                        row.append("")
                        continue
                    r = ff.get(col[0], {}).get(s, {}).get(k, {})
                    row += [num(r.get(x)) for x in ("AD", "MA", "Both", "Neither")]
                w.writerow(row)


def archive(dest):
    """Copy the small result files of this run (no .npy matrices) into dest, keeping the folder layout."""
    patterns = ["check_uniform_output.txt", "run_all_log*.txt", "run_times*.txt"]
    for m in P.METHODS:
        folder = P.METHOD_DIRS[m]
        patterns += [f"{folder}/*.png"]
        sub = f"{folder}/results/smoke" if SMOKE else f"{folder}/results"
        patterns += [f"{sub}/*.json", f"{sub}/*.png", f"{folder}/results/*{'_smoke' if SMOKE else '_output'}.txt"]
    hsub = "6.hybrid/results/smoke" if SMOKE else "6.hybrid/results"
    patterns += [f"{hsub}/*.json", f"{hsub}/*.png", f"{hsub}/*/*.png", "6.hybrid/results/*.txt"]
    if not SMOKE:
        patterns += ["CTI_eval/CTI_Eval_*.json", "CTI_eval/CTI_summary.*", "CTI_eval/CTI_eval_output.txt",
                     "CTI_eval/CTI_method_ranks.npz"]
    n = 0
    for pat in patterns:
        for src in glob.glob(os.path.join(HERE, pat)):
            rel = os.path.relpath(src, HERE)
            if rel.endswith("CTI_Eval_empty_template.json"):
                continue
            os.makedirs(os.path.join(dest, os.path.dirname(rel)), exist_ok=True)
            shutil.copy2(src, os.path.join(dest, rel))
            n += 1
    return n


def main():
    attr, ff, cti, missing = gather()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(HERE, "results_archive", ("smoke_" if SMOKE else "") + stamp)
    os.makedirs(dest, exist_ok=True)
    text = text_report(attr, ff, cti, missing)
    with open(os.path.join(dest, "summary.txt"), "w") as fh:
        fh.write(text)
    excel_blocks(attr, ff, cti, os.path.join(dest, "excel_blocks.csv"))
    n = archive(dest)
    print(text)
    print(f"Saved {os.path.relpath(dest, HERE)}/summary.txt and excel_blocks.csv (+ {n} result files copied)")


if __name__ == "__main__":
    main()
